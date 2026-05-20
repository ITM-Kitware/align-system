from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from align_system.utils import logging
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.types import Action as MCTSAction, ToolSpec
from align_system.interfaces.ai2thor_interface import AI2ThorAction
from align_system.utils import logging
from align_system.data_models.dialog import DialogElement
from align_system.algorithms.lib.persona.types import Action, Observation, ToolSpec, PlanCandidate
from align_system.utils.alignment_utils import AvgDistScalarAlignment
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class Node:
    parent: Optional["Node"]
    # Instead of prior_action, store the macro that this edge represents:
    plan_actions: List[Action] = field(default_factory=list)

    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    unexpanded: List[PlanCandidate] = field(default_factory=list)
    rationale: str = ""       # proposer's rationale for this candidate
    critic_reason: str = ""   # critic's explanation of its score

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

class MCTSPlannerADMComponent(ADMComponent):
    """
    Full MCTS planner ADM component.
    Runs `expansions` MCTS iterations using a structured inference engine
    for both proposer (candidate generation) and critic (rollout scoring).
    Returns the best planned action sequence as AI2ThorAction objects.
    """

    _PROPOSER_SCHEMA = (
        '{"candidates":['
        '{"actions":[{"tool_name":"MoveAhead","args":{"moveMagnitude":0.25}}],'
        '"rationale":"..."}]}'
    )
    _CRITIC_SCHEMA = '{"score":0.75,"reason":"..."}'

    def __init__(
        self,
        structured_inference_engine,
        expansions: int = 5,
        proposals_per_expand: int = 4,
        rollout_horizon: int = 4,
        uct_c: float = 1.4,
        proposer_temperature: float = 0.7,
        critic_temperature: float = 0.2,
        target_kdma_names_override: list = None,
        attributes: dict = None,
        regression_component=None,
        alignment_function=None,
    ):
        self.structured_inference_engine = structured_inference_engine
        self.expansions = expansions
        self.proposals_per_expand = proposals_per_expand
        self.rollout_horizon = rollout_horizon
        self.uct_c = uct_c
        self.proposer_temperature = proposer_temperature
        self.critic_temperature = critic_temperature
        self.target_kdma_names_override = target_kdma_names_override
        self.attributes = attributes or {}
        self.regression_component = regression_component
        if regression_component is not None and alignment_function is None:
            self.alignment_function = AvgDistScalarAlignment()
        else:
            self.alignment_function = alignment_function
        self._history: List[MCTSAction] = []

    # ------------------------------------------------------------------
    # History management (called by PipelineADM)
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        self._history = []

    def update_history(self, chosen_action) -> None:
        if chosen_action is None:
            return
        plan = getattr(chosen_action, "plan", None)
        if plan:
            self._history.extend(plan)
        else:
            tool_name = (
                chosen_action.action_id
                if hasattr(chosen_action, "action_id")
                else str(chosen_action)
            )
            args = getattr(chosen_action, "args", {}) or {}
            self._history.append(MCTSAction(tool_name=tool_name, args=args))

    # ------------------------------------------------------------------
    # ADMComponent interface
    # ------------------------------------------------------------------

    def run_returns(self):
        return "chosen_action"

    def run(self, scenario_state, actions: List[AI2ThorAction], alignment_target=None) -> AI2ThorAction:
        tool_map = {a.action_id: a for a in actions}
        tools = [
            ToolSpec(
                name=a.action_id,
                description=a.unstructured,
                json_schema={"type": "object", "properties": {}, "required": []},
            )
            for a in actions
        ]
        obs = Observation(text=getattr(scenario_state, "unstructured", ""))
        task = getattr(scenario_state, "unstructured", "")

        best_plan = self._plan(task=task, obs=obs, tools=tools, action_history=list(self._history), alignment_target=alignment_target, scenario_state=scenario_state)

        if not best_plan:
            log.warning("[MCTSPlanner] no plan found; falling back to first action")
            return actions[0]

        for mcts_action in best_plan:
            if mcts_action.tool_name in tool_map:
                orig = tool_map[mcts_action.tool_name]
                return AI2ThorAction(
                    action_id=mcts_action.tool_name,
                    unstructured=orig.unstructured,
                    args=mcts_action.args or {},
                    plan=best_plan,
                )
            log.warning(f"[MCTSPlanner] unknown tool '{mcts_action.tool_name}', skipping")

        return actions[0]

    # ------------------------------------------------------------------
    # LLM calls (proposer + critic)
    # ------------------------------------------------------------------

    def _propose(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[MCTSAction],
    ) -> List[PlanCandidate]:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        history_lines = (
            "\n".join(f"- {a.tool_name}({a.args})" for a in action_history)
            if action_history else "None"
        )
        system_prompt = (
            "You are an embodied planning model.\n"
            "Return ONLY valid JSON. No extra text.\n"
            f"Generate {self.proposals_per_expand} semi-diverse candidate plans."
        )
        user_prompt = (
            f"Task: {task}\n\n"
            f"Observation:\n{obs.text}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            f"Action history:\n{history_lines}\n\n"
            f"Generate {self.proposals_per_expand} diverse candidate plans.\n"
            f"- Each plan is 1 to {self.rollout_horizon} actions.\n"
            f"- Use ONLY the tool names provided.\n"
            "- Vary tool choice; include at least one exploration move.\n"
            "- IMPORTANT objectId rule: For tools requiring objectId, copy the exact full objectId "
            "from the observation (values after 'id='). Never use object type names as objectId.\n"
            "- Avoid repeating the same last action unless clearly helpful."
        )
        dialog = [
            DialogElement(content=system_prompt, role="system"),
            DialogElement(content=user_prompt, role="user"),
        ]
        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
        log.info("[bold]*PROMPT FOR PROPOSER*[/bold]", extra={"markup": True})
        log.info(dialog_prompt)
        response = self.structured_inference_engine.run_inference(
            [dialog_prompt], self._PROPOSER_SCHEMA, temperature=self.proposer_temperature
        )[0]
        raw_candidates = response.get("candidates", []) if isinstance(response, dict) else []
        log.info(f"[MCTSPlanner] proposed {len(raw_candidates)} candidates")

        candidates: List[PlanCandidate] = []
        for cand in raw_candidates:
            raw_actions = cand.get("actions", []) if isinstance(cand, dict) else []
            rationale = (cand.get("rationale", "") if isinstance(cand, dict) else "").strip()
            plan_actions = [
                MCTSAction(
                    tool_name=a.get("tool_name", "") if isinstance(a, dict) else a.tool_name,
                    args=a.get("args") or {} if isinstance(a, dict) else a.args or {},
                )
                for a in raw_actions
                if (a.get("tool_name") if isinstance(a, dict) else getattr(a, "tool_name", None))
            ]
            if plan_actions:
                candidates.append(PlanCandidate(actions=plan_actions, rationale=rationale))
        return candidates

    def _score(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[MCTSAction],
        proposed_future: List[MCTSAction],
    ) -> Tuple[float, str]:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        history_lines = (
            "\n".join(f"- {a.tool_name}({a.args})" for a in action_history)
            if action_history else "None"
        )
        future_lines = "\n".join(f"- {a.tool_name}({a.args})" for a in proposed_future)
        system_prompt = (
            "You are a critic evaluating proposed action sequences.\n"
            "Return ONLY valid JSON with a score (0.0–1.0) and a brief reason."
        )
        user_prompt = (
            f"Task: {task}\n\n"
            f"Observation:\n{obs.text}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            f"Action history:\n{history_lines}\n\n"
            f"Proposed future actions:\n{future_lines}\n\n"
            "Score 0.0–1.0: how likely is this sequence to make progress toward the task?"
        )
        dialog = [
            DialogElement(content=system_prompt, role="system"),
            DialogElement(content=user_prompt, role="user"),
        ]
        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
        log.info("[bold]*PROMPT FOR CRITIC*[/bold]", extra={"markup": True})
        log.info(dialog_prompt)
        response = self.structured_inference_engine.run_inference(
            [dialog_prompt], self._CRITIC_SCHEMA, temperature=self.critic_temperature
        )[0]
        score = 0.0
        reason = ""
        if isinstance(response, dict):
            score = float(response.get("score", 0.0))
            reason = str(response.get("reason", ""))
        return score, reason

    def _resolve_target_kdmas(self, alignment_target) -> List[dict]:
        """Extract and filter target KDMAs from alignment_target, applying override if set."""
        target_kdmas = []
        if alignment_target is not None:
            raw = getattr(alignment_target, "kdma_values", []) or []
            for k in raw:
                d = k.to_dict() if hasattr(k, "to_dict") else dict(k)
                target_kdmas.append(d)

        if alignment_target is not None and self.target_kdma_names_override is not None:
            from_target = {k["kdma"]: k for k in target_kdmas}
            overridden = []
            for name in self.target_kdma_names_override:
                if name == "*":
                    overridden.extend(target_kdmas)
                elif name in from_target:
                    overridden.append(from_target[name])
                elif name in self.attributes:
                    attr = self.attributes[name]
                    overridden.append({"kdma": attr.kdma, "value": None})
            target_kdmas = overridden

        return target_kdmas

    def _candidates_to_keys(self, candidates: List[PlanCandidate]) -> List[str]:
        """Format candidates as short, JSON-safe keys for use as choice identifiers.

        Keys use only alphanumeric + underscore so the LLM can reproduce them
        exactly as JSON property names. No args or special chars are included.
        """
        keys = []
        for i, cand in enumerate(candidates or []):
            tool_chain = "_then_".join(a.tool_name for a in (cand.actions or []) if a.tool_name)
            keys.append(f"cand{i}_{tool_chain}" if tool_chain else f"cand{i}_empty")

        # Deduplicate: if two candidates have identical tool chains, append a counter
        seen: Dict[str, int] = {}
        unique_keys = []
        for key in keys:
            if key in seen:
                seen[key] += 1
                unique_keys.append(f"{key}_{seen[key]}")
            else:
                seen[key] = 0
                unique_keys.append(key)
        return unique_keys

    def _score_with_regression(
        self,
        scenario_state,
        candidates: List[PlanCandidate],
        alignment_target,
    ) -> Dict[int, Tuple[float, str]]:
        """Score candidates using the regression component + alignment function.

        Uses ComparativeRegressionADMComponent to predict per-candidate KDMA scores,
        then applies AvgDistScalarAlignment (or direct formula for null targets) to
        compute a backprop score in [0, 1].
        """
        candidate_keys = self._candidates_to_keys(candidates)

        # Determine which attributes will each trigger a separate inference call
        attr_override = getattr(self.regression_component, "target_attribute_names_override", None)
        if attr_override is not None:
            from align_system.utils.alignment_utils import attributes_in_alignment_target
            base_attrs = (
                attributes_in_alignment_target(alignment_target)
                if alignment_target is not None else []
            )
            attr_names = [n for n in attr_override if n != "*"]
            if "*" in attr_override:
                attr_names = attr_names + base_attrs
        else:
            attr_names = list(getattr(self.regression_component, "attributes", {}).keys())

        log.info(
            f"[bold]*REGRESSION CRITIC: {len(candidate_keys)} candidates "
            f"× {len(attr_names)} attribute call(s): {attr_names}*[/bold]",
            extra={"markup": True},
        )
        for i, key in enumerate(candidate_keys):
            log.info(f"  [bold]candidate_{i}:[/bold] {key}", extra={"markup": True})

        reasonings, pred_scores, _ = self.regression_component.run(
            scenario_state=scenario_state,
            choices=candidate_keys,
            alignment_target=alignment_target,
        )

        target_kdmas = self._resolve_target_kdmas(alignment_target)
        if not target_kdmas:
            return {i: (0.5, "") for i in range(len(candidates))}

        # Collect per-candidate reasoning strings
        reasons: Dict[int, str] = {}
        for i, key in enumerate(candidate_keys):
            cand_reasonings = reasonings.get(key, {})
            parts = [f"{k}: {rlist[0]}" for k, rlist in cand_reasonings.items() if rlist]
            reasons[i] = "; ".join(parts)

        # When all targets are scalar, delegate to alignment_function for normalized scores
        all_scalar = all(k.get("value") is not None for k in target_kdmas)
        if all_scalar and self.alignment_function is not None:
            target_kdma_keys = {k["kdma"] for k in target_kdmas}
            filtered_scores = {
                key: {kdma: vals for kdma, vals in scores.items() if kdma in target_kdma_keys}
                for key, scores in pred_scores.items()
            }
            try:
                _, probs_dict = self.alignment_function(filtered_scores, target_kdmas)
                results = {
                    i: (probs_dict.get(key, 0.0), reasons[i])
                    for i, key in enumerate(candidate_keys)
                }
                log.info(
                    f"[MCTSPlanner] regression+alignment scores: "
                    f"{ {k: f'{v[0]:.3f}' for k, v in results.items()} }"
                )
                return results
            except Exception as e:
                log.warning(
                    f"[MCTSPlanner] alignment_function failed ({e}), using direct scoring"
                )

        # Direct scoring: 1 - |target - avg| for scalar targets, avg for null targets
        results = {}
        for i, key in enumerate(candidate_keys):
            cand_scores = pred_scores.get(key, {})
            contributions = []
            for k in target_kdmas:
                kdma_key = k["kdma"]
                samples = cand_scores.get(kdma_key, [0.5])
                if not isinstance(samples, list):
                    samples = [samples]
                avg_score = sum(samples) / len(samples) if samples else 0.5
                target_value = k.get("value")
                if target_value is not None:
                    contributions.append(max(0.0, 1.0 - abs(target_value - avg_score)))
                else:
                    contributions.append(avg_score)
            backprop_score = sum(contributions) / len(contributions) if contributions else 0.5
            results[i] = (backprop_score, reasons[i])

        log.info(
            f"[MCTSPlanner] regression scores: "
            f"{ {k: f'{v[0]:.3f}' for k, v in results.items()} }"
        )
        return results

    def _score_comparative(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[MCTSAction],
        candidates: List[PlanCandidate],
        alignment_target,
        scenario_state=None,
    ) -> Dict[int, Tuple[float, str]]:
        if self.regression_component is not None and scenario_state is not None and alignment_target is not None:
            return self._score_with_regression(scenario_state, candidates, alignment_target)

        # No alignment target — score each candidate independently
        results = {}
        for i, cand in enumerate(candidates):
            future = (cand.actions or [])[:self.rollout_horizon]
            s, r = self._score(task, obs, tools, action_history, future)
            results[i] = (s, r)
        return results

    # ------------------------------------------------------------------
    # MCTS internals
    # ------------------------------------------------------------------

    def _uct_select(self, node: Node) -> Node:
        assert node.children
        log_n = math.log(node.visits + 1)

        def uct(n: Node) -> float:
            if n.visits == 0:
                return float("inf")
            return n.value + self.uct_c * math.sqrt(log_n / n.visits)

        return max(node.children, key=uct)

    def _reconstruct_actions(self, node: Node) -> List[MCTSAction]:
        chunks: List[List[MCTSAction]] = []
        cur = node
        while cur and cur.parent is not None:
            chunks.append(cur.plan_actions)
            cur = cur.parent
        chunks.reverse()
        out: List[MCTSAction] = []
        for ch in chunks:
            out.extend(ch)
        return out

    def _plan(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[MCTSAction],
        alignment_target=None,
        scenario_state=None,
    ) -> List[MCTSAction]:
        root = Node(parent=None, plan_actions=[])

        for _ in range(self.expansions):
            # 1) selection
            node = root
            while node.children and not node.unexpanded:
                node = self._uct_select(node)

            prefix = self._reconstruct_actions(node)
            log.info(
                f"[MCTSPlanner] expand node={'root' if node.parent is None else 'child'} "
                f"depth={len(prefix)} prefix={[a.tool_name for a in prefix]}"
            )

            # 2) expansion
            if not node.unexpanded:
                node.unexpanded = self._propose(
                    task=task,
                    obs=obs,
                    tools=tools,
                    action_history=action_history + prefix,
                )
                log.info(f"[MCTSPlanner] proposed paths: {node.unexpanded}")

                # Score all siblings comparatively in one critic call
                if node.unexpanded:
                    comparative_scores = self._score_comparative(
                        task=task,
                        obs=obs,
                        tools=tools,
                        action_history=action_history + prefix,
                        candidates=node.unexpanded,
                        alignment_target=alignment_target,
                        scenario_state=scenario_state,
                    )
                    for idx, cand in enumerate(node.unexpanded):
                        cand.score, cand.critic_reason = comparative_scores.get(idx, (0.0, ""))

            if not node.unexpanded:
                continue

            cand = node.unexpanded.pop(0)
            if not cand.actions:
                continue

            child = Node(parent=node, plan_actions=cand.actions, rationale=cand.rationale)
            node.children.append(child)

            # 3) use pre-computed comparative score
            score = cand.score if cand.score is not None else 0.0
            critic_reason = cand.critic_reason
            action_strs = ", ".join(f"{a.tool_name}({a.args})" for a in (cand.actions or []))
            log.info(f"[EXPAND] rationale: {cand.rationale}")
            log.info(f"[EXPAND] actions:   {action_strs}")
            child.critic_reason = critic_reason
            log.info(f"[SCORE]  {score:.3f}  reason: {critic_reason}")

            # 4) backprop
            cur = child
            while cur is not None:
                cur.visits += 1
                cur.value_sum += score
                cur = cur.parent

        if not root.children:
            return []

        self._print_tree(root)
        best = max(root.children, key=lambda n: n.value)
        cur = best
        while cur.children:
            cur = max(cur.children, key=lambda n: n.value)
        return self._reconstruct_actions(cur)

    def _print_tree(self, root: Node) -> None:
        all_values = [n.value for n in self._all_nodes(root) if n.visits > 0 and n.parent is not None]
        lo = min(all_values) if all_values else 0.0
        hi = max(all_values) if all_values else 1.0
        mid = (lo + hi) / 2

        def _colour(node: Node) -> str:
            if node.visits == 0:
                return "\033[90m"
            if hi == lo:
                return "\033[92m"
            if node.value >= mid + (hi - mid) * 0.5:
                return "\033[92m"
            elif node.value >= mid:
                return "\033[93m"
            else:
                return "\033[91m"

        def _node_label(node: Node) -> str:
            if node.parent is None:
                return "[ROOT]"
            action_strs = [
                f"{a.tool_name}({a.args})" if hasattr(a, "tool_name") else str(a)
                for a in (node.plan_actions or [])
            ]
            actions_display = ", ".join(action_strs) if action_strs else "(empty)"
            value_str = f"{node.value:.3f}" if node.visits > 0 else "unvisited"
            return f"[v={value_str} | n={node.visits}] → {actions_display}"

        def _draw(node: Node, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            colour, reset = _colour(node), "\033[0m"
            log.info(f"{prefix}{connector}{colour}{_node_label(node)}{reset}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                _draw(child, child_prefix, i == len(node.children) - 1)

        log.info("\n" + "═" * 60)
        log.info("  MCTS TREE  (after expansion)")
        log.info("═" * 60)
        log.info(f"\033[97m[ROOT]\033[0m  visits={root.visits}")
        for i, child in enumerate(root.children):
            _draw(child, "", i == len(root.children) - 1)

        if root.children:
            path, cur = [], max(root.children, key=lambda n: n.value)
            while cur is not None and cur.parent is not None:
                path.append(cur)
                cur = max(cur.children, key=lambda n: n.value) if cur.children else None

            log.info("\n" + "─" * 60)
            log.info("  \033[92m★ Best path\033[0m")
            for depth, n in enumerate(path):
                indent = "  " + "    " * depth
                actions = ", ".join(
                    f"{a.tool_name}({a.args})" if hasattr(a, "tool_name") else str(a)
                    for a in (n.plan_actions or [])
                )
                log.info(f"{indent}↳ [v={n.value:.3f} | n={n.visits}]  {actions}")
                if n.rationale:
                    log.info(f"{indent}  proposer: {n.rationale}")
                if n.critic_reason:
                    log.info(f"{indent}  critic:   {n.critic_reason}")
        log.info("═" * 60 + "\n")

    def _all_nodes(self, root: Node) -> list:
        result, stack = [], [root]
        while stack:
            n = stack.pop()
            result.append(n)
            stack.extend(n.children)
        return result
