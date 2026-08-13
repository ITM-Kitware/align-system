import json
from collections import defaultdict
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from align_system.algorithms.abstracts import ActionBasedADM
from align_system.algorithms.open_world_components import (
    OWFormatChoicesADMComponent,
)
from align_system.data_models.dialog import DialogElement
from align_system.prompt_engineering.outlines_prompts import (
    DefaultITMBaselineSystemPrompt,
    Phase2ScenarioDescriptionWCasualtyInfo,
    action_choice_json_schema,
)
from align_system.utils import call_with_coerced_args, logging

log = logging.getLogger(__name__)


class AgentGraphState(TypedDict, total=False):
    scenario_state: Any
    actions: list
    history: list

    observation: str
    choices: list
    choice_to_action_mapping: dict

    proposal: dict
    feedback: list
    attempts: int

    chosen_action: Any
    proposal_history: list


class LangGraphAgentADM(ActionBasedADM):
    """Observe -> propose -> act agent ADM built on LangGraph.

    Replaces the comparative regression + midpoint alignment pipeline
    with a single agent loop: the agent observes the scenario, proposes
    an action, and the proposal is validated/completed before being
    returned to the driver.  Invalid proposals are retried with the
    validation error fed back into the prompt.  KDMA/alignment targets
    are not used; decisions are driven purely by the scenario and the
    system prompt.
    """

    def __init__(self,
                 structured_inference_engine,
                 choice_to_action_component,
                 action_parameter_completion_component,
                 scenario_description_template=None,
                 system_prompt=None,
                 max_retries: int = 3,
                 reasoning_max_length: int = 512,
                 max_history_in_prompt: int = 10):
        self.structured_inference_engine = structured_inference_engine
        self.choice_to_action_component = choice_to_action_component
        self.action_parameter_completion_component =\
            action_parameter_completion_component

        if scenario_description_template is None:
            scenario_description_template = Phase2ScenarioDescriptionWCasualtyInfo()
        self.scenario_description_template = scenario_description_template

        if system_prompt is None:
            system_prompt = DefaultITMBaselineSystemPrompt()
        self.system_prompt = system_prompt

        self.max_retries = max_retries
        self.reasoning_max_length = reasoning_max_length
        self.max_history_in_prompt = max_history_in_prompt

        # Per-scenario log of decisions already taken, so the agent
        # can recall its own history across an unfolding scenario
        self.decision_log = defaultdict(list)

        self.format_choices_component = OWFormatChoicesADMComponent()

        graph = StateGraph(AgentGraphState)
        graph.add_node("observe", self._observe)
        graph.add_node("propose", self._propose)
        graph.add_node("validate", self._validate)
        graph.add_node("fallback", self._fallback)
        graph.add_edge(START, "observe")
        graph.add_edge("observe", "propose")
        graph.add_edge("propose", "validate")
        graph.add_conditional_edges(
            "validate",
            self._after_validate,
            {"done": END, "retry": "propose", "give_up": "fallback"})
        graph.add_edge("fallback", END)

        self.graph = graph.compile()

    def _observe(self, state: AgentGraphState) -> dict:
        choices, choice_to_action_mapping = self.format_choices_component.run(
            state['scenario_state'], state['actions'])

        scenario_description = call_with_coerced_args(
            self.scenario_description_template,
            {'scenario_state': state['scenario_state']})

        return {'observation': scenario_description,
                'choices': choices,
                'choice_to_action_mapping': choice_to_action_mapping}

    def _propose(self, state: AgentGraphState) -> dict:
        system_content = self.system_prompt()

        choices_block = "\n".join(f"- {c}" for c in state['choices'])
        user_content = (f"Scenario:\n{state['observation']}\n\n"
                        f"Possible responses:\n{choices_block}\n")

        history = state.get('history', [])
        if history:
            history_block = "\n".join(
                f"{i}. {entry}" for i, entry in enumerate(history, start=1))
            user_content += ("\nActions you have already taken in this "
                             f"scenario, in order:\n{history_block}\n")

        user_content += ("\nGiven the scenario, reason about the best "
                         "response and select it from the possible "
                         "responses.")

        feedback = state.get('feedback', [])
        if feedback:
            feedback_block = "\n".join(f"- {f}" for f in feedback)
            user_content += ("\n\nYour previous selection(s) could not "
                             f"be carried out:\n{feedback_block}\n"
                             "Please select a different response.")

        dialog = [DialogElement(role='system', content=system_content),
                  DialogElement(role='user', content=user_content)]

        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
        log.info("[bold]*AGENT PROPOSE PROMPT*[/bold]", extra={"markup": True})
        log.info(dialog_prompt)

        proposal = self.structured_inference_engine.run_inference(
            dialog_prompt,
            action_choice_json_schema(
                json.dumps(state['choices']), self.reasoning_max_length))

        log.info("[bold]*AGENT PROPOSAL*[/bold]", extra={"markup": True})
        log.info(json.dumps(proposal, indent=2))

        proposal_history = [*state.get('proposal_history', []), proposal]

        return {'proposal': proposal,
                'proposal_history': proposal_history,
                'attempts': state.get('attempts', 0) + 1}

    def _validate(self, state: AgentGraphState) -> dict:
        proposal = state['proposal']

        try:
            chosen_action, _ = self.choice_to_action_component.run(
                state['scenario_state'],
                proposal['action_choice'],
                state['choice_to_action_mapping'],
                justification=proposal.get('detailed_reasoning'))

            chosen_action, _ = self.action_parameter_completion_component.run(
                state['scenario_state'], chosen_action)
        except Exception as e:
            log.warning(f"Agent proposal failed validation: {e}")
            return {'feedback': [*state.get('feedback', []),
                                 f"{proposal['action_choice']}: {e}"],
                    'chosen_action': None}

        return {'chosen_action': chosen_action}

    def _after_validate(self, state: AgentGraphState) -> str:
        if state.get('chosen_action') is not None:
            return "done"
        elif state.get('attempts', 0) >= self.max_retries:
            return "give_up"
        else:
            return "retry"

    def _fallback(self, state: AgentGraphState) -> dict:
        # Couldn't produce a valid proposal within max_retries; take
        # the first available action so the live session can proceed
        log.warning("Agent failed to produce a valid action after "
                    f"{state.get('attempts', 0)} attempts; falling back "
                    "to the first available action")

        chosen_action = state['actions'][0]
        if (hasattr(chosen_action, 'justification')
                and chosen_action.justification is None):
            chosen_action.justification =\
                "Fallback selection after repeated invalid proposals"

        return {'chosen_action': chosen_action}

    def _log_decision(self, scenario_id, chosen_action):
        description = (getattr(chosen_action, 'unstructured', None)
                       or str(getattr(chosen_action, 'action_type', 'action')))

        entry = description
        justification = getattr(chosen_action, 'justification', None)
        if justification:
            entry += f" (because: {justification[:200]})"

        self.decision_log[scenario_id].append(entry)

    def choose_action(self,
                      scenario_state,
                      available_actions,
                      alignment_target=None,
                      scenario_id=None,
                      **kwargs):
        # alignment_target is accepted for interface compatibility but
        # deliberately unused; this ADM does not align to KDMA targets
        history = self.decision_log[scenario_id][-self.max_history_in_prompt:]

        final_state = self.graph.invoke(
            {'scenario_state': scenario_state,
             'actions': available_actions,
             'history': history,
             'feedback': [],
             'attempts': 0})

        self._log_decision(scenario_id, final_state['chosen_action'])

        # Values of choice_info are expected to be dicts (see
        # per-choice handling in the drivers)
        choice_info = {
            'agent': {'observation': final_state.get('observation'),
                      'history': history,
                      'choices': final_state.get('choices'),
                      'proposal_history': final_state.get('proposal_history', []),
                      'feedback': final_state.get('feedback', []),
                      'attempts': final_state.get('attempts', 0)}}

        return final_state['chosen_action'], choice_info
