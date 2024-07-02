import json

import outlines
import jinja2
from rich.highlighter import JSONHighlighter
from swagger_client.models import (
    ActionTypeEnum,
    InjuryLocationEnum,
    CharacterTagEnum,
)

from align_system.utils import logging
from align_system.utils import get_swagger_class_enum_values
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.prompt_engineering.outlines_prompts import (
    baseline_system_prompt,
    high_moral_deservingness_system_prompt,
    low_moral_deservingness_system_prompt,
    high_maximization_system_prompt,
    low_maximization_system_prompt,
    action_selection_prompt,
    scenario_state_description_1,
    followup_clarify_character,
    followup_clarify_treatment,
    followup_clarify_tag,
    action_choice_json_schema,
    character_choice_json_schema,
    tag_choice_json_schema,
    treatment_choice_json_schema,
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class OutlinesTransformersADM(ActionBasedADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 baseline=False,
                 **kwargs):
        self.baseline = baseline
        self.model = outlines.models.transformers(
            model_name,
            device=device,
            model_kwargs=kwargs.get('model_kwargs', {}),
            tokenizer_kwargs=kwargs.get('tokenizer_kwargs', {}))

    def dialog_to_prompt(self, dialog):
        tokenizer = self.model.tokenizer.tokenizer

        try:
            encoded_dialog = tokenizer.apply_chat_template(dialog)
        except jinja2.exceptions.TemplateError:
            # Assume that the tokenizer chat template doesn't accept
            # system messages; combine system message first user
            # message
            system_msg, user_msg, *rest = dialog

            assert user_msg['role'] == 'user'

            updated_content = system_msg['content'] + '\n' + user_msg['content']

            dialog = [{'role': 'user', 'content': updated_content}, *rest]

            encoded_dialog = tokenizer.apply_chat_template(dialog)

        return tokenizer.decode(encoded_dialog)

    @staticmethod
    def kdma_value_to_system_prompt(kdma, value):
        if kdma == "MoralDesert":
            if value < 0.5:
                return low_moral_deservingness_system_prompt()
            else:
                return high_moral_deservingness_system_prompt()
        elif kdma == "maximization":
            if value < 0.5:
                return low_maximization_system_prompt()
            else:
                return high_maximization_system_prompt()
        else:
            return None

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        # choices = ["({}) {}".format(chr(i + 65), a.unstructured)
        #            for i, a in enumerate(available_actions)]
        choices = [a.unstructured for a in available_actions]

        scenario_description = scenario_state_description_1(scenario_state)

        prompt = action_selection_prompt(scenario_description, choices)

        if not self.baseline and alignment_target is not None:
            kdma_values = alignment_target.kdma_values

            if len(kdma_values) != 1:
                raise RuntimeError("This ADM assumes a single KDMA target, aborting!")

            kdma = kdma_values[0]['kdma']
            value = kdma_values[0]['value']

            system_prompt = self.__class__.kdma_value_to_system_prompt(kdma, value)

            if system_prompt is None:
                log.warning("Couldn't find system prompt for kdma: {}, and "
                            "value: {}.  Using baseline system prompt".format(
                                kdma, value))
                system_prompt = baseline_system_prompt()
        else:
            system_prompt = baseline_system_prompt()

        dialog = [{'role': 'system', 'content': system_prompt},
                  {'role': 'user', 'content': prompt}]
        dialog_text = self.dialog_to_prompt(dialog)

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        generator = outlines.generate.json(
            self.model,
            action_choice_json_schema(json.dumps(choices)),
            whitespace_pattern=r"[ ]?")

        log.info("[bold]*DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(dialog_text)

        selected_choice = generator(dialog_text)
        selected_choice_idx = choices.index(selected_choice['action_choice'])

        log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(selected_choice, extra={"highlighter": JSON_HIGHLIGHTER})

        action_to_take = available_actions[selected_choice_idx]
        action_to_take.justification = selected_choice['detailed_reasoning']

        if action_to_take.action_type in {ActionTypeEnum.APPLY_TREATMENT,
                                          ActionTypeEnum.TAG_CHARACTER,
                                          ActionTypeEnum.CHECK_ALL_VITALS,
                                          ActionTypeEnum.CHECK_PULSE,
                                          ActionTypeEnum.CHECK_RESPIRATION,
                                          ActionTypeEnum.MOVE_TO_EVAC}:
            dialog.append({'role': 'assistant',
                           'content': '{}  I would choose to {}'.format(
                               selected_choice['detailed_reasoning'],
                               selected_choice['action_choice'])})
            dialog.append({'role': 'user',
                           'content': followup_clarify_character(scenario_state)})
            dialog_text = self.dialog_to_prompt(dialog)

            characters = [c.name for c in scenario_state.characters]

            generator = outlines.generate.json(
                self.model,
                character_choice_json_schema(json.dumps(characters)),
                whitespace_pattern=r"[ ]?")

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_character = generator(dialog_text)
            selected_character_idx = characters.index(selected_character['character_choice'])

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_character, extra={"highlighter": JSON_HIGHLIGHTER})

            action_to_take.character_id = scenario_state.characters[selected_character_idx].id

        if action_to_take.action_type == ActionTypeEnum.APPLY_TREATMENT:
            valid_treatment_locations = get_swagger_class_enum_values(InjuryLocationEnum)
            available_supplies = [s for s in scenario_state.supplies if s.quantity > 0]

            dialog.append({'role': 'assistant',
                           'content': '{}  {} should receive the action.'.format(
                               selected_character['brief_reasoning'],
                               selected_character['character_choice'])})
            dialog.append({'role': 'user',
                           'content': followup_clarify_treatment(
                               scenario_state.characters[selected_character_idx],
                               available_supplies)})

            dialog_text = self.dialog_to_prompt(dialog)

            generator = outlines.generate.json(
                self.model,
                treatment_choice_json_schema(
                    json.dumps([s.type for s in available_supplies]),
                    json.dumps(valid_treatment_locations)),
                whitespace_pattern=r"[ ]?")

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_treatment = generator(dialog_text)

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_treatment, extra={"highlighter": JSON_HIGHLIGHTER})

            if action_to_take.parameters is None:
                action_to_take.parameters = {}

            action_to_take.parameters['treatment'] = selected_treatment['supplies_to_use']
            action_to_take.parameters['location'] = selected_treatment['treatment_location']

        if action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            valid_tags = get_swagger_class_enum_values(CharacterTagEnum)

            dialog.append({'role': 'assistant',
                           'content': '{}  {} should receive the action.'.format(
                               selected_character['brief_reasoning'],
                               selected_character['character_choice'])})

            selected_character_dict =\
                scenario_state.characters[selected_character_idx].to_dict()
            dialog.append({'role': 'user',
                           'content': followup_clarify_tag(
                               selected_character_dict)})

            dialog_text = self.dialog_to_prompt(dialog)

            generator = outlines.generate.json(
                self.model,
                tag_choice_json_schema(
                    json.dumps(valid_tags)),
                whitespace_pattern=r"[ ]?")

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_tag = generator(dialog_text)

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_tag, extra={"highlighter": JSON_HIGHLIGHTER})

            if action_to_take.parameters is None:
                action_to_take.parameters = {}

            action_to_take.parameters['category'] = selected_tag['triage_tag']

        return action_to_take
