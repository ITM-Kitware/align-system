from __future__ import annotations

from typing import List, Optional
from align_system.utils import logging
from align_system.algorithms.abstracts import ADMComponent
#from align_system.algorithms.planner_adm.llm_ollama import OllamaAI2ThorProposer, OllamaConfig
from align_system.data_models.types import Action as PlannerAction, ToolSpec
from align_system.interfaces.ai2thor_interface import AI2ThorAction
from align_system.utils import logging
from align_system.data_models.dialog import DialogElement
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
        num_candidates: int = 3,
        rollout_horizon: int = 3,
        inference_temperature: Optional[float] = None,
    ):
        self.structured_inference_engine = structured_inference_engine
        self.num_candidates = num_candidates
        self.rollout_horizon = rollout_horizon
        self.inference_temperature = inference_temperature
        self._history: List[PlannerAction] = []
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

        # candidates = self.proposer.propose(
        #     task=scenario_state.unstructured,
        #     obs=obs,
        #     tools=tools,
        #     action_history=self._history,
        #     k=self.num_candidates,
        #     diversity_hint="Vary tool choice; include at least one exploration move.",
        # )

        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        history_lines = (
            "\n".join(f"- {a.tool_name}({a.args})" for a in self._history)
            if self._history else "None"
        )
        predict_proposer_prompt = (
            f"Task: {scenario_state.unstructured}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            f"Action history:\n{history_lines}\n\n"
            f"Generate {self.num_candidates} diverse candidate plans."
        )

        score_schema = (
            '{"candidates":[{"actions":[{"tool_name":"MoveAhead","args":{"moveMagnitude":0.25}}],'
            '"rationale":"..."}]}'
        )

        prompt_system = ("You are an embodied planning model.\n"
            "Return ONLY valid JSON. No extra text.\n"
            f"Generate {self.num_candidates} semi-diverse candidate plans.\n")        
        prompt = (
            f"You are an embodied planning model.\n"
            "Return ONLY valid JSON. No extra text.\n"
            f"Generate {self.num_candidates} diverse candidate plans.\n"
            f"- Each plan is 1 to {self.rollout_horizon} actions.\n"
            f"- Use ONLY the tool names provided.\n"
            f"- Args MUST satisfy each tool schema.\n"
            f"- IMPORTANT objectId rule: For tools requiring objectId (TeleportNearObject, PickupObject, "
            f"OpenObject, CloseObject, ToggleObjectOn/Off), you MUST copy the exact full objectId string "
            "from the observation's visible lines (the value after 'id='). "
            "Never use object type names like 'Apple' as objectId. Full objectIds contain '|' characters.\n"
            "- Avoid repeating the same last action unless clearly helpful.\n"
        )


        dialog = []
        dialog.insert(0,DialogElement(content=prompt_system, role="system"))
        dialog.append(DialogElement(content=prompt, role="user"))
        dialog.append(DialogElement(content=predict_proposer_prompt, role="user"))
        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
        log.info("[bold]*PROMPT FOR PROPOSER*[/bold]",
                    extra={"markup": True})
        log.info(dialog_prompt)
        response = self.structured_inference_engine.run_inference(
            [dialog_prompt], score_schema, temperature=0.7)[0]
        candidates = response.get("candidates", []) if isinstance(response, dict) else []

        log.info(f"[PlannerCandidateGenerator] proposed {len(candidates)} candidates")

        candidate_actions: List[AI2ThorAction] = []
        seen: set = set()

        for cand in candidates[: self.num_candidates]:
            cand_actions = cand.get("actions", []) if isinstance(cand, dict) else []
            if not cand_actions:
                continue
            planner_action = cand_actions[0]
            if isinstance(planner_action, dict):
                tool_name = planner_action.get("tool_name", "")
            elif isinstance(planner_action, str):
                tool_name = planner_action
            else:
                tool_name = planner_action.tool_name

            if tool_name not in tool_map:
                log.warning(f"[PlannerCandidateGenerator] unknown tool '{tool_name}', skipping")
                continue

            if isinstance(planner_action, dict):
                args = planner_action.get("args") or {}
            elif isinstance(planner_action, str):
                args = {}
            else:
                args = planner_action.args or {}

            dedup_key = (tool_name, frozenset((k, str(v)) for k, v in args.items()))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            rationale = (cand.get("rationale", "") if isinstance(cand, dict) else cand.rationale).strip()

            plan = []
            for a in cand_actions:
                if isinstance(a, dict):
                    plan.append(PlannerAction(tool_name=a.get("tool_name", ""), args=a.get("args") or {}))
                elif isinstance(a, str):
                    plan.append(PlannerAction(tool_name=a, args={}))
                else:
                    plan.append(PlannerAction(tool_name=a.tool_name, args=a.args or {}))
            action_sequence = " -> ".join(a.tool_name for a in plan)
            label = f"{action_sequence}: {rationale[:80]}" if rationale else action_sequence
            candidate_actions.append(
                AI2ThorAction(
                    action_id=tool_name,
                    unstructured=label,
                    args=args,
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
