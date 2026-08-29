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

    Action history is provided by `PipelineADM` via the injected
    `history` argument: a list of prior working_output dicts, each
    annotated (by the driver, via `PipelineADM.update_history()`) with
    an `executed_action` recording what actually succeeded in the
    environment and `failed_actions` recording attempts that failed.
    Failed attempts are surfaced to the LLM so it can avoid repeating
    them.
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

    @staticmethod
    def _action_history_from_pipeline_history(history):
        """Flatten the pipeline's working_output history into two lists of
        `PlannerAction`s: actions that succeeded in the environment and
        attempts that failed."""
        executed_history: List[PlannerAction] = []
        failed_history: List[PlannerAction] = []
        for entry in history or []:
            executed = entry.get("executed_action")
            if executed is not None:
                plan = getattr(executed, "plan", None)
                if plan:
                    executed_history.extend(plan)
                else:
                    tool_name = (
                        executed.action_id
                        if hasattr(executed, "action_id")
                        else str(executed)
                    )
                    args = getattr(executed, "args", {}) or {}
                    executed_history.append(PlannerAction(tool_name=tool_name, args=args))
            failed_history.extend(entry.get("failed_actions") or [])
        return executed_history, failed_history

    def run_returns(self):
        return "actions"

    def run(self, scenario_state, actions: List[AI2ThorAction],
            history=None) -> List[AI2ThorAction]:
        tool_map = {a.action_id: a for a in actions}

        tools = [
            ToolSpec(
                name=a.action_id,
                description=a.unstructured,
                json_schema=(getattr(a, "tool_schema", None)
                             or {"type": "object", "properties": {}, "required": []}),
            )
            for a in actions
        ]

        executed_history, failed_history = \
            self._action_history_from_pipeline_history(history)
        template_args = {
            "scenario_state": scenario_state,
            "tools": tools,
            "action_history": executed_history,
            "failed_attempts": failed_history,
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

        # Downstream components key choices by the unstructured label;
        # make sure duplicates are disambiguated
        label_counts: dict = {}
        for action in candidate_actions:
            n = label_counts.get(action.unstructured, 0) + 1
            label_counts[action.unstructured] = n
            if n > 1:
                action.unstructured += f" (alternative {n})"

        log.info(
            "[PlannerCandidateGenerator] candidates: "
            + ", ".join(a.action_id for a in candidate_actions)
        )
        return candidate_actions
