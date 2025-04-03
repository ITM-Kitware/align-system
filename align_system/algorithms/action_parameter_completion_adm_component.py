import copy
import json

from rich.highlighter import JSONHighlighter
from swagger_client.models import (
    ActionTypeEnum,
    InjuryLocationEnum,
    CharacterTagEnum
)

from align_system.utils import logging
from align_system.utils import adm_utils
from align_system.utils import get_swagger_class_enum_values
from align_system.algorithms.abstracts import ADMComponent
from align_system.prompt_engineering.outlines_prompts import (
    action_selection_prompt,
    scenario_state_description_1,
    followup_clarify_aid,
    followup_clarify_character,
    followup_clarify_treatment,
    followup_clarify_treatment_from_list,
    followup_clarify_tag,
    aid_choice_json_schema,
    character_choice_json_schema,
    tag_choice_json_schema,
    treatment_choice_json_schema,
    treatment_choice_from_list_json_schema)
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class ActionParameterCompletionADMComponent(ADMComponent):
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

            dialog.append(DialogElement(role='user',
                                        content=prompt,
                                        namespace='.',
                                        tags=['parameter_completion']))

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
        if action_to_take.action_type in {ActionTypeEnum.APPLY_TREATMENT,
                                          ActionTypeEnum.TAG_CHARACTER,
                                          ActionTypeEnum.CHECK_ALL_VITALS,
                                          ActionTypeEnum.CHECK_PULSE,
                                          ActionTypeEnum.CHECK_RESPIRATION,
                                          ActionTypeEnum.CHECK_BLOOD_OXYGEN,
                                          ActionTypeEnum.MOVE_TO_EVAC,
                                          ActionTypeEnum.MOVE_TO}:
            action_to_take, selected_character, selected_character_idx, dialog =\
                self.ensure_character_id_is_populated(scenario_state, action_to_take, dialog)

        if action_to_take.action_type == ActionTypeEnum.APPLY_TREATMENT:
            if action_to_take.parameters is None or 'treatment' not in action_to_take.parameters or 'location' not in action_to_take.parameters:
                # TODO: Add inference kwarg to use heurustic treatment options or not
                from align_system.algorithms.apply_treatment import treatment_options

                character_injuries = [i.to_dict() for i in scenario_state.characters[selected_character_idx].injuries]
                supplies = [s.to_dict() for s in scenario_state.supplies]

                heuristic_treatment_options = treatment_options(character_injuries, supplies)

                # Filter heuristic treatment options by already
                # populated treatment or location
                att_treatment = None
                att_location = None
                if action_to_take.parameters is not None:
                    att_treatment = action_to_take.parameters.get('treatment')
                    att_location = action_to_take.parameters.get('location')

                filtered_heuristic_treatments = []
                filtered_heuristic_params = []
                for heuristic_treatment, heuristic_params in zip(heuristic_treatment_options.get('treatments', ()),
                                                                 heuristic_treatment_options.get('parameters', ())):
                    if att_treatment is not None and heuristic_params['treatment'] != att_treatment:
                        continue
                    if att_location is not None and heuristic_params['location'] != att_location:
                        continue

                    filtered_heuristic_treatments.append(heuristic_treatment)
                    filtered_heuristic_params.append(heuristic_params)

                filtered_heuristic_treatment_options = copy.deepcopy(heuristic_treatment_options)
                filtered_heuristic_treatment_options['treatments'] = filtered_heuristic_treatments
                filtered_heuristic_treatment_options['parameters'] = filtered_heuristic_params

                # Should fall back to subsequent treatment / location
                # handler if no heuristic treatment options left after
                # filtering.
                if len(filtered_heuristic_treatments) > 0:
                    log.debug("[bold]*FILTERED HEURISTIC TREATMENT OPTIONS*[/bold]",
                              extra={"markup": True})
                    log.debug(filtered_heuristic_treatment_options)
                    action_to_take, selected_treatment, dialog =\
                        self.select_treatment_parameters(scenario_state,
                                                         action_to_take,
                                                         selected_character,
                                                         selected_character_idx,
                                                         dialog,
                                                         filtered_heuristic_treatment_options)
                else:
                    log.debug("[bold]*NO FILTERED HEURISTIC TREATMENT OPTIONS*[/bold]")

            # Use follow up prompt to define treatment and/or location if neccesary
            if action_to_take.parameters is None or 'treatment' not in action_to_take.parameters or 'location' not in action_to_take.parameters:
                action_to_take, selected_treatment, dialog =\
                    self.populate_treatment_parameters(scenario_state,
                                                       action_to_take,
                                                       selected_character,
                                                       selected_character_idx,
                                                       dialog)
        elif action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            if action_to_take.parameters is None or 'category' not in action_to_take.parameters:
                action_to_take, selected_tag, dialog =\
                    self.populate_tagging_parameters(scenario_state,
                                                     action_to_take,
                                                     selected_character,
                                                     selected_character_idx,
                                                     dialog)
        # Set aid_id for MOVE_TO_EVAC if missing
        elif action_to_take.action_type == ActionTypeEnum.MOVE_TO_EVAC:
            if action_to_take.parameters is None or "aid_id" not in action_to_take.parameters:
                action_to_take, selected_aid, dialog =\
                    self.populate_aid_parameters(scenario_state,
                                                 action_to_take,
                                                 selected_character,
                                                 selected_character_idx,
                                                 dialog)

        return action_to_take, dialog

    def ensure_character_id_is_populated(self,
                                         scenario_state,
                                         action_to_take,
                                         dialog):
        if action_to_take.character_id is None:
            # Use follow up prompt to define selected_character
            if action_to_take.action_type not in {ActionTypeEnum.MOVE_TO_EVAC,
                                                  ActionTypeEnum.MOVE_TO}:
                characters = [c for c in scenario_state.characters if not c.unseen]

            if action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
                # Further filtering for tagging action, don't tag
                # a character that already has a tag
                characters = [c for c in characters if c.tag is None]
            elif action_to_take.action_type in {ActionTypeEnum.CHECK_ALL_VITALS,
                                                ActionTypeEnum.CHECK_PULSE,
                                                ActionTypeEnum.CHECK_RESPIRATION,
                                                ActionTypeEnum.CHECK_BLOOD_OXYGEN}:
                # Further filtering for assessment actions, don't
                # allow an already "visited" character to be assessed
                # again; NOTE: Not certain this won't prevent us from
                # doing legitimate actions in some corner cases
                characters = [c for c in characters
                              if c.visited is None or not c.visited]

            dialog.append(DialogElement(role='assistant',
                                        content='{}  I would choose to {}'.format(
                                            action_to_take.justification,
                                            action_to_take.unstructured),
                                        namespace='.',
                                        tags=['parameter_completion']))
            dialog.append(DialogElement(role='user',
                                        content=followup_clarify_character(characters),
                                        namespace='.',
                                        tags=['parameter_completion']))
            dialog_text = self.structured_inference_engine.dialog_to_prompt(dialog)

            character_names = [c.name for c in characters]

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_character = self.structured_inference_engine.run_inference(
                dialog_text,
                character_choice_json_schema(json.dumps(character_names)))

            selected_character_idx = character_names.index(selected_character['character_choice'])

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_character, extra={"highlighter": JSON_HIGHLIGHTER})

            action_to_take.character_id = characters[selected_character_idx].id
        else:
            # Use action_to_take.character_id to define selected_character
            selected_character = {}
            for char_index in range(len(scenario_state.characters)):
                character = scenario_state.characters[char_index]
                if character.id == action_to_take.character_id:
                    selected_character['character_choice'] = character.name
                    selected_character_idx = char_index
                    break

            selected_character['brief_reasoning'] = action_to_take.justification

        return action_to_take, selected_character, selected_character_idx, dialog

    def populate_treatment_parameters(self,
                                      scenario_state,
                                      action_to_take,
                                      selected_character,
                                      selected_character_idx,
                                      dialog):
        # Get valid injury locations for the selected character
        valid_treatment_locations = []
        for injury in scenario_state.characters[selected_character_idx].injuries:
            valid_treatment_locations.append(injury.location)

        # Work-around when injuries are not yet discovered (intend actions)
        if len(valid_treatment_locations) == 0:
            log.info("No injuries on selected character. Allowing any treatment location")
            valid_treatment_locations = get_swagger_class_enum_values(InjuryLocationEnum)

        # If there is only one treatment location and we have the treatment, we don't need a follow-up
        if len(valid_treatment_locations) == 1 and action_to_take.parameters is not None and 'treatment' in action_to_take.parameters:
            action_to_take.parameters['location'] = valid_treatment_locations[0]

            selected_treatment = {'detailed_reasoning': '<Treatment already specified and only one valid treatment location>',
                                  'supplies_to_use': action_to_take.parameters['treatment'],
                                  'treatment_location': action_to_take.parameters['location']}
        # If there are multiple treatment locations and/or we are missing the treatment, use follow-up
        else:
            available_supplies = [s for s in scenario_state.supplies if s.quantity > 0]

            dialog.append(DialogElement(role='assistant',
                                        content='{}  {} should receive the action.'.format(
                                            selected_character['brief_reasoning'],
                                            selected_character['character_choice']),
                                        namespace='.',
                                        tags=['parameter_completion']))
            dialog.append(DialogElement(role='user',
                                        content=followup_clarify_treatment(
                                            scenario_state.characters[selected_character_idx],
                                            available_supplies),
                                        namespace='.',
                                        tags=['parameter_completion']))

            dialog_text = self.structured_inference_engine.dialog_to_prompt(dialog)

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_treatment = self.structured_inference_engine(
                [dialog_text],
                treatment_choice_json_schema(
                    json.dumps([s.type for s in available_supplies]),
                    json.dumps(valid_treatment_locations)))

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_treatment, extra={"highlighter": JSON_HIGHLIGHTER})

            # Use follow-up response to define only the missing fields
            if action_to_take.parameters is None:
                action_to_take.parameters = {}
            if 'treatment' not in action_to_take.parameters:
                action_to_take.parameters['treatment'] = selected_treatment['supplies_to_use']
            if 'location' not in action_to_take.parameters:
                action_to_take.parameters['location'] = selected_treatment['treatment_location']

        return action_to_take, selected_treatment, dialog

    def select_treatment_parameters(self,
                                    scenario_state,
                                    action_to_take,
                                    selected_character,
                                    selected_character_idx,
                                    dialog,
                                    heuristic_treatment_options):
        possible_treatments = heuristic_treatment_options['treatments']

        # If there is only one treatment location and we have the
        # treatment, we don't need a follow-up
        if len(possible_treatments) == 0:
            #  TODO: Handle this case prior to calling this function
            raise RuntimeError("No possible treatments from heuristic_treatment_options!")
        elif len(possible_treatments) == 1:
            log.debug("[bold]*SELECTING ONLY REMAINING HEURISTIC TREATMENT OPTION*[/bold]")

            # Assumes correspondence between 'treatments' and 'parameters'
            assert len(heuristic_treatment_options['parameters']) == 1

            treatment_parameters = heuristic_treatment_options['parameters'][0]
            selected_treatment = {'detailed_reasoning': '<Only one heuristic treatment option available>',
                                  'supplies_to_use': treatment_parameters['treatment'],
                                  'treatment_location': treatment_parameters['location']}
        # If there are multiple treatment locations and/or we are missing the treatment, use follow-up
        else:
            available_supplies = [s for s in scenario_state.supplies if s.quantity > 0]

            dialog.append(DialogElement(role='assistant',
                                        content='{}  {} should receive the action.'.format(
                                            selected_character['brief_reasoning'],
                                            selected_character['character_choice']),
                                        namespace='.',
                                        tags=['parameter_completion']))
            dialog.append(DialogElement(role='user',
                                        content=followup_clarify_treatment_from_list(
                                            scenario_state.characters[selected_character_idx],
                                            available_supplies,
                                            possible_treatments),
                                        namespace='.',
                                        tags=['parameter_completion']))

            dialog_text = self.structured_inference_engine.dialog_to_prompt(dialog)

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_treatment = self.structured_inference_engine.run_inference(
                dialog_text,
                treatment_choice_from_list_json_schema(
                    json.dumps(possible_treatments)))

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_treatment, extra={"highlighter": JSON_HIGHLIGHTER})

            treatment_idx = possible_treatments.index(selected_treatment['treatment_choice'])
            treatment_parameters = heuristic_treatment_options['parameters'][treatment_idx]

        # Use follow-up response to define only the missing fields
        if action_to_take.parameters is None:
            action_to_take.parameters = {}

        action_to_take.parameters = {**action_to_take.parameters, **treatment_parameters}

        return action_to_take, selected_treatment, dialog

    def populate_tagging_parameters(self,
                                    scenario_state,
                                    action_to_take,
                                    selected_character,
                                    selected_character_idx,
                                    dialog):
        valid_tags = get_swagger_class_enum_values(CharacterTagEnum)

        dialog.append(DialogElement(role='assistant',
                                    content='{}  {} should receive the action.'.format(
                                        selected_character['brief_reasoning'],
                                        selected_character['character_choice']),
                                    namespace='.',
                                    tags=['parameter_completion']))

        selected_character_dict =\
            scenario_state.characters[selected_character_idx].to_dict()
        dialog.append(DialogElement(role='user',
                                    content=followup_clarify_tag(
                                        selected_character_dict),
                                    namespace='.',
                                    tags=['parameter_completion']))

        dialog_text = self.structured_inference_engine.dialog_to_prompt(dialog)

        log.info("[bold]*DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(dialog_text)

        selected_tag = self.structured_inference_engine.run_inference(
            dialog_text,
            tag_choice_json_schema(
                json.dumps(valid_tags)))

        log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(selected_tag, extra={"highlighter": JSON_HIGHLIGHTER})

        if action_to_take.parameters is None:
            action_to_take.parameters = {}

        action_to_take.parameters['category'] = selected_tag['triage_tag']

        return action_to_take, selected_tag, dialog

    def populate_aid_parameters(self,
                                scenario_state,
                                action_to_take,
                                selected_character,
                                selected_character_idx,
                                dialog):
        selected_character_dict =\
            scenario_state.characters[selected_character_idx].to_dict()

        # Limit to the aids that will accept the selected patient
        available_aids = [
            aid
            for aid in scenario_state.environment.decision_environment.aid
            if (
                aid.patients_treated is None or
                "military_disposition" not in selected_character_dict or
                selected_character_dict["miliary_disposition"] in aid.patients_treated
            )
        ]

        if len(available_aids) == 0:
            raise RuntimeError("No aids to choose from")
        elif len(available_aids) == 1:  # If there is only one option, we don't need a follow-up
            action_to_take.parameters["aid_id"] = available_aids[0].id

            selected_aid = {'brief_reasoning': '<Only one aid option available>',
                            'aid_choice': action_to_take.parameters["aid_id"]}
        else:
            dialog.append(DialogElement(role='assistant',
                                        content='{}  {} should receive the action.'.format(
                                            selected_character['brief_reasoning'],
                                            selected_character['character_choice']),
                                        namespace='.',
                                        tags=['parameter_completion']))
            dialog.append(DialogElement(role='user',
                                        content=followup_clarify_aid(
                                            selected_character_dict,
                                            available_aids),
                                        namespace='.',
                                        tags=['parameter_completion']))

            dialog_text = self.structured_inference_engine.dialog_to_prompt(dialog)

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_aid = self.structured_inference_engine.run_inference(
                dialog_text,
                aid_choice_json_schema(
                    json.dumps([aid.id for aid in available_aids])))

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_aid, extra={"highlighter": JSON_HIGHLIGHTER})

            action_to_take.parameters["aid_id"] = selected_aid['aid_choice']

        return action_to_take, selected_aid, dialog
