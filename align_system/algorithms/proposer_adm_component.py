from __future__ import annotations

from typing import List, Optional

from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.ai2thor import Action as PlannerAction, ToolSpec
from align_system.data_models.dialog import DialogElement
from align_system.interfaces.ai2thor_interface import AI2ThorAction
from align_system.utils import call_with_coerced_args, logging

log = logging.getLogger(__name__)


class ProposerGeneratorAgent(ADMComponent):
    """
    Pipeline step that narrows the full AI2Thor tool set down to a
    small number of semantically motivated candidates before handing
    off to comparative regression.

    The planner proposer LLM is called once per step to generate
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
        structured_inference_engine,
        prompt_template,
        schema_template,
        system_prompt_template=None,
        num_candidates: int = 3,
        rollout_horizon: int = 3,
        inference_temperature: Optional[float] = None,
    ):
        self.structured_inference_engine = structured_inference_engine
        self.prompt_template = prompt_template
        self.schema_template = schema_template
        self.system_prompt_template = system_prompt_template
        self.num_candidates = num_candidates
        self.rollout_horizon = rollout_horizon
        self.inference_temperature = inference_temperature
        self._history: List[PlannerAction] = []

    # ------------------------------------------------------------------
    # History management (called by PipelineADM)
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        self._history = []

    def update_history(self, chosen_action) -> None:
        """Record the action chosen by downstream alignment so the next
        proposal round knows what was already tried."""
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
            self._history.append(PlannerAction(tool_name=tool_name, args=args))

    # ------------------------------------------------------------------
    # ADMComponent interface
    # ------------------------------------------------------------------

    def run_returns(self):
        return "actions"

    def run(self, scenario_state, actions: List[AI2ThorAction]) -> List[AI2ThorAction]:
        tool_map = {a.action_id: a for a in actions}

        tools = [
            ToolSpec(
                name=a.action_id,
                description=a.unstructured,
                json_schema={"type": "object", "properties": {}, "required": []},
            )
            for a in actions
        ]

        template_args = {
            "scenario_state": scenario_state,
            "tools": tools,
            "action_history": self._history,
            "num_candidates": self.num_candidates,
            "rollout_horizon": self.rollout_horizon,
        }

        dialog = []
        if self.system_prompt_template is not None:
            system_prompt = call_with_coerced_args(
                self.system_prompt_template, template_args)
            dialog.append(DialogElement(role="system", content=system_prompt))

        propose_prompt = call_with_coerced_args(
            self.prompt_template, template_args)
        dialog.append(DialogElement(role="user", content=propose_prompt))

        candidates_schema = call_with_coerced_args(
            self.schema_template, template_args)

        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
        log.info("[bold]*PROMPT FOR PROPOSER*[/bold]",
                 extra={"markup": True})
        log.info(dialog_prompt)
        response = self.structured_inference_engine.run_inference(
            [dialog_prompt],
            candidates_schema,
            temperature=self.inference_temperature)[0]
        candidates = response.get("candidates", []) if isinstance(response, dict) else []

        log.info(f"[PlannerCandidateGenerator] proposed {len(candidates)} candidates")

        candidate_actions: List[AI2ThorAction] = []
        seen: set = set()

        for cand in candidates[: self.num_candidates]:
            cand_actions = cand.get("actions", []) if isinstance(cand, dict) else []
            if not cand_actions:
                continue

            plan = [
                PlannerAction(tool_name=a.get("tool_name", ""), args=a.get("args") or {})
                for a in cand_actions
            ]
            first_action = plan[0]

            if first_action.tool_name not in tool_map:
                log.warning(f"[PlannerCandidateGenerator] unknown tool "
                            f"'{first_action.tool_name}', skipping")
                continue

            dedup_key = (first_action.tool_name,
                         frozenset((k, str(v)) for k, v in first_action.args.items()))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            rationale = (cand.get("rationale") or "").strip()

            action_sequence = " -> ".join(a.tool_name for a in plan)
            label = f"{action_sequence}: {rationale[:80]}" if rationale else action_sequence
            candidate_actions.append(
                AI2ThorAction(
                    action_id=first_action.tool_name,
                    unstructured=label,
                    args=first_action.args,
                    justification=rationale,
                    plan=plan,
                )
            )

        # Fallback: if proposer returned nothing useful, use first N actions
        if not candidate_actions:
            log.warning("[PlannerCandidateGenerator] no valid candidates; falling back to first N actions")
            candidate_actions = [
                AI2ThorAction(
                    action_id=a.action_id,
                    unstructured=a.unstructured,
                    args={},
                )
                for a in actions[: self.num_candidates]
            ]

        log.info(
            "[PlannerCandidateGenerator] candidates: "
            + ", ".join(a.action_id for a in candidate_actions)
        )
        return candidate_actions
