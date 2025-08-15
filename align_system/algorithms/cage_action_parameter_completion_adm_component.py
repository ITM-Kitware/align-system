import copy
import json

from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.utils import adm_utils
from align_system.algorithms.abstracts import ADMComponent
from align_system.prompt_engineering.outlines_prompts import (
        action_selection_prompt,
        scenario_state_description_1,
        followup_clarify_hostnames_cage,
        cage_hostname_choice_json_schema,
    )
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class CAGEActionParameterCompletionADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine):
        self.structured_inference_engine = structured_inference_engine

    # TODO: Copied from outlines_adm.py; should use a common template/prompt
    def _state_to_top_level_prompt(self, scenario_state, actions):
        """
        Generate prompt dialog based on given state and actions
        """
        choices = adm_utils.format_choices(
            [a.unstructured for a in actions],
            actions,
            scenario_state
        )

        scenario_description = scenario_state_description_1(scenario_state)
        prompt = action_selection_prompt(scenario_description, choices)

        return prompt, choices

    def run_returns(self):
        return ('chosen_action',
                'action_parameter_completion_dialog')

    def run(self,
            scenario_state,
            actions,
            choices,
            chosen_choice,
            dialog=None,
            alignment_target=None):
        if dialog is None:
            # If prior steps didn't provide any dialog/context, use a
            # sensible default:
            prompt, _ = self._state_to_top_level_prompt(
                scenario_state,
                actions)

            dialog = [DialogElement(role='user',
                                    content=prompt,
                                    tags=['parameter_completion'])]

        # If last dialog message is an 'assistant' message, remove it
        # as we'll generate one for each follow-up needed.  (Dialogs
        # should have alternating assistant/user elements)
        if dialog[-1].role == 'assistant':
            dialog.pop()

        chosen_choice_idx = choices.index(chosen_choice)
        chosen_action = actions[chosen_choice_idx]

        action_to_take, output_dialog = self.populate_action_parameters(
                scenario_state, chosen_action, dialog)

        return action_to_take, output_dialog

    def populate_action_parameters(self, scenario_state, action_to_take, dialog):
        if action_to_take.name in { 'Analyse', 'Misinform', 'Remove', 'Restore'}:
            action_to_take, selected_hostname, selected_hostname_idx, dialog =\
                self.ensure_hostname_is_populated(scenario_state, action_to_take, dialog)


        return action_to_take, dialog

    def ensure_hostname_is_populated(self,
                                         scenario_state,
                                         action_to_take,
                                         dialog):
        if action_to_take.hostname is None:
            # Use follow up prompt to define selected_character
            hostnames = [c for c in scenario_state.hostnames ]


            dialog.append(DialogElement(role='assistant',
                                        content='{}  I would choose to {}'.format(
                                            action_to_take.justification,
                                            action_to_take.unstructured),
                                        tags=['parameter_completion']))
            dialog.append(DialogElement(role='user',
                                        content=followup_clarify_hostnames_cage(hostnames),
                                        tags=['parameter_completion']))
            dialog_text = self.structured_inference_engine.dialog_to_prompt(dialog)


            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_hostname = self.structured_inference_engine.run_inference(
                dialog_text,
                cage_hostname_choice_json_schema(json.dumps(hostnames)))

            selected_hostname_idx = hostnames.index(selected_hostname['hostname_choice'])

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_hostname, extra={"highlighter": JSON_HIGHLIGHTER})

            action_to_take.hostname = hostnames[selected_hostname_idx]
        else:
            # Use action_to_take.hostname_id to define selected_hostname
            selected_hostname = {}
            for char_index in range(len(scenario_state.hostnames)):
                hostname = scenario_state.hostnames[char_index]
                if hostname == action_to_take.hostname:
                    selected_hostname['hostname_choice'] = hostname
                    selected_hostname_idx = char_index
                    break

            selected_hostname['brief_reasoning'] = action_to_take.justification

        return action_to_take, selected_hostname, selected_hostname_idx, dialog

