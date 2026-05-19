from __future__ import annotations

import math
from typing import List, Optional, Tuple
from align_system.utils import logging
from align_system.algorithms.abstracts import ADMComponent
#from align_system.algorithms.mcts_adm.llm_ollama import OllamaAI2ThorProposer, OllamaConfig
from align_system.data_models.types import Action as MCTSAction, ToolSpec
from align_system.interfaces.ai2thor_interface import AI2ThorAction
from align_system.utils import logging
from align_system.data_models.dialog import DialogElement
from align_system.algorithms.lib.persona.types import Action, Observation, ToolSpec, PlanCandidate
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
    ):
        self.structured_inference_engine = structured_inference_engine
        self.expansions = expansions
        self.proposals_per_expand = proposals_per_expand
        self.rollout_horizon = rollout_horizon
        self.uct_c = uct_c
        self.proposer_temperature = proposer_temperature
        self.critic_temperature = critic_temperature
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

    def run(self, scenario_state, actions: List[AI2ThorAction]) -> AI2ThorAction:
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

        best_plan = self._plan(task=task, obs=obs, tools=tools, action_history=list(self._history))

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
    ) -> List[MCTSAction]:
        root = Node(parent=None, plan_actions=[])

        for _ in range(self.expansions):
            # 1) selection
            node = root
            while node.children and not node.unexpanded:
                node = self._uct_select(node)

            prefix = self._reconstruct_actions(node)

            # 2) expansion
            if not node.unexpanded:
                node.unexpanded = self._propose(
                    task=task,
                    obs=obs,
                    tools=tools,
                    action_history=action_history + prefix,
                )
                log.info(f"[MCTSPlanner] proposed paths: {node.unexpanded}")

            if not node.unexpanded:
                continue

            cand = node.unexpanded.pop(0)
            if not cand.actions:
                continue

            child = Node(parent=node, plan_actions=cand.actions, rationale=cand.rationale)
            node.children.append(child)

            # 3) rollout scoring (critic)
            future = (cand.actions or [])[:self.rollout_horizon]
            action_strs = ", ".join(f"{a.tool_name}({a.args})" for a in (cand.actions or []))
            log.info(f"[EXPAND] rationale: {cand.rationale}")
            log.info(f"[EXPAND] actions:   {action_strs}")
            score, critic_reason = self._score(
                task=task,
                obs=obs,
                tools=tools,
                action_history=action_history + prefix,
                proposed_future=future,
            )
            child.critic_reason = critic_reason
            log.info(f"[SCORE]  {score:.3f}  reason: {critic_reason}")
            if score is None:
                score = 0.0

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
