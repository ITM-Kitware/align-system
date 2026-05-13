from __future__ import annotations

from typing import List, Optional

from align_system.algorithms.abstracts import ADMComponent
from align_system.algorithms.mcts_adm.llm_ollama import OllamaAI2ThorProposer, OllamaConfig
from align_system.algorithms.mcts_adm.types import Action as MCTSAction, Observation, ToolSpec
from align_system.interfaces.ai2thor_interface import AI2ThorAction
from align_system.utils import logging

log = logging.getLogger(__name__)


class MCTSCandidateGeneratorADMComponent(ADMComponent):
    """
    Pipeline step that narrows the full AI2Thor tool set down to a
    small number of semantically motivated candidates before handing
    off to comparative regression.

    The MCTS proposer LLM is called once per step to generate
    `num_candidates` candidate actions with rationales.  The rationale
    is embedded in the action's `unstructured` field so that
    comparative regression sees it as part of the choice description.

    Action history is maintained across `run()` calls within a scenario
    and cleared by `reset_history()`.  After comparative regression
    picks a winner, `update_history()` should be called so future
    proposals avoid repeating the same action.
    """

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        temperature: float = 0.7,
        num_ctx: int = 8192,
        num_candidates: int = 3,
        rollout_horizon: int = 3,
    ):
        cfg = OllamaConfig(
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            max_actions_per_plan=rollout_horizon,
        )
        self.proposer = OllamaAI2ThorProposer(cfg)
        self.num_candidates = num_candidates
        self._history: List[MCTSAction] = []
        self._pending_tool: Optional[str] = None

    # ------------------------------------------------------------------
    # History management (called by PipelineADM)
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        self._history = []
        self._pending_tool = None

    def update_history(self, chosen_action) -> None:
        """Record the action chosen by downstream alignment so the next
        proposal round knows what was already tried."""
        if chosen_action is not None:
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
        return "actions"

    def run(self, scenario_state, actions: List[AI2ThorAction]) -> List[AI2ThorAction]:
        obs = Observation(text=scenario_state.unstructured)
        tool_map = {a.action_id: a for a in actions}

        tools = [
            ToolSpec(
                name=a.action_id,
                description=a.unstructured,
                json_schema={"type": "object", "properties": {}, "required": []},
            )
            for a in actions
        ]

        candidates = self.proposer.propose(
            task=scenario_state.unstructured,
            obs=obs,
            tools=tools,
            action_history=self._history,
            k=self.num_candidates,
            diversity_hint="Vary tool choice; include at least one exploration move.",
        )

        log.info(f"[MCTSCandidateGenerator] proposed {len(candidates)} candidates")

        candidate_actions: List[AI2ThorAction] = []
        seen: set = set()

        for cand in candidates[: self.num_candidates]:
            if not cand.actions:
                continue
            mcts_action = cand.actions[0]
            tool_name = mcts_action.tool_name

            if tool_name not in tool_map:
                log.warning(f"[MCTSCandidateGenerator] unknown tool '{tool_name}', skipping")
                continue

            # Deduplicate by (tool_name, frozenset of args)
            dedup_key = (tool_name, frozenset((k, str(v)) for k, v in (mcts_action.args or {}).items()))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Embed the rationale in `unstructured` so comparative regression
            # sees it as part of the choice description.
            rationale = cand.rationale.strip()
            label = f"{tool_name}: {rationale[:80]}" if rationale else tool_name

            candidate_actions.append(
                AI2ThorAction(
                    action_id=tool_name,
                    unstructured=label,
                    args=mcts_action.args or {},
                    justification=rationale,
                    plan=list(cand.actions),
                )
            )

        # Fallback: if proposer returned nothing useful, use first N actions
        if not candidate_actions:
            log.warning("[MCTSCandidateGenerator] no valid candidates; falling back to first N actions")
            candidate_actions = [
                AI2ThorAction(
                    action_id=a.action_id,
                    unstructured=a.unstructured,
                    args={},
                )
                for a in actions[: self.num_candidates]
            ]

        log.info(
            "[MCTSCandidateGenerator] candidates: "
            + ", ".join(a.action_id for a in candidate_actions)
        )
        return candidate_actions
