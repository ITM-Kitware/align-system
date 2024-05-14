import json

import outlines
import jinja2
from rich.highlighter import JSONHighlighter
from swagger_client.models import ActionTypeEnum, InjuryStatusEnum, InjuryLocationEnum, CharacterTagEnum

from align_system.utils import logging
from align_system.utils import get_swagger_class_enum_values
from align_system.algorithms.abstracts import ActionBasedADM

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


@outlines.prompt
def baseline_system_prompt():
    """
    You are an assistant specialized in answering multiple-choice \
    questions related to medical triage. When a question is posed, \
    carefully analyze the symptoms or conditions \
    described. Respond by providing a detailed reasoning using a \
    step-by-step process or process of elimination.  Conclude with \
    the final answer, represented by the corresponding index \
    number. Your response should be grounded in established \
    medical knowledge and aim to be informative. \
    """


@outlines.prompt
def action_selection_prompt(situation, choices):
    """
    {{ situation }}

    Given the following possible actions, please provide a detailed \
    reasoning for what action you would take:
    {% for choice in choices %}
    {{ choice }}
    {% endfor %}
    """


@outlines.prompt
def scenario_state_description_1(scenario_state):
    """
    CHARACTERS:
    {% for character in scenario_state.characters %}
    {{ character.name }}: {{ character.unstructured }}
    {{ character.name }}'s intent: {{ character.intent }}

    {% endfor %}

    SITUATION:
    {{ scenario_state.unstructured }}

    """


@outlines.prompt
def followup_clarify_character(scenario_state):
    """
    Please clarify which character should receive the action.

    CHARACTERS:
    {% for character in scenario_state.characters %}
    {{ character.name }}: {{ character.unstructured }}
    {{ character.name }}'s intent: {{ character.intent }}

    {% endfor %}
    """


@outlines.prompt
def followup_clarify_treatment(character,
                               supplies,
                               treated_value=InjuryStatusEnum.TREATED):
    """
    Given the casualty description below, please state which injury \
    should be treated with the supplies available, and where the \
    treatment should be applied

    {{ character.unstructured }}
    INJURIES:
    {% for injury in character.injuries %}
    {% if not injury.status == treated_value %}
    {"injury_name": {{ injury.name }}, "injury_severity": {{ injury.severity }}, "injury_location": {{ injury.location }} }
    {% endif %}
    {% endfor %}

    AVAILABLE SUPPLIES:
    {% for supply in supplies %}
    {{ supply.quantity }} x {{ supply.type }}
    {% endfor %}
    """


@outlines.prompt
def followup_clarify_tag(character,
                         treated_value=InjuryStatusEnum.TREATED):
    """
    Given the casualty description below, please state what triage tag \
    should be applied to the casualty given their injuries and vitals

    {{ character.unstructured }}
    INJURIES:
    {% for injury in character.injuries %}
    {% if not injury.status == treated_value %}
    {"injury_name": {{ injury.name }}, "injury_severity": {{ injury.severity }}, "injury_location": {{ injury.location }} }
    {% endif %}
    {% endfor %}

    VITALS:
    {% for name, value in character.vitals.items() %}
    {% if value is not none %}
    {{ name }}: {{ value }}
    {% endif %}
    {% endfor %}
    """


@outlines.prompt
def action_choice_json_schema(choices_json_str):
    '''
    {"$defs": {"ActionChoice": {"enum": {{ choices_json_str }},
       "title": "ActionChoice",
       "type": "string"}},
     "properties": {"detailed_reasoning": {"title": "Detailed Reasoning",
       "type": "string"},
      "action_choice": {"$ref": "#/$defs/ActionChoice"}},
     "required": ["detailed_reasoning", "action_choice"],
     "title": "ActionSelection",
     "type": "object"}
    '''


@outlines.prompt
def character_choice_json_schema(choices_json_str):
    '''
    {"$defs": {"CharacterChoice": {"enum": {{ choices_json_str }},
       "title": "CharacterChoice",
       "type": "string"}},
     "properties": {"brief_reasoning": {"title": "Brief Reasoning",
       "type": "string"},
      "character_choice": {"$ref": "#/$defs/CharacterChoice"}},
     "required": ["brief_reasoning", "character_choice"],
     "title": "CharacterSelection",
     "type": "object"}
    '''


@outlines.prompt
def tag_choice_json_schema(tags_json_str):
    '''
    {"$defs": {"TriageTag": {"enum": {{ tags_json_str }},
       "title": "TriageTag",
       "type": "string"}},
     "properties": {"detailed_reasoning": {"title": "Detailed Reasoning",
       "type": "string"},
      "triage_tag": {"$ref": "#/$defs/TriageTag"}},
     "required": ["detailed_reasoning", "triage_tag"],
     "title": "TagSelection",
     "type": "object"}
    '''


@outlines.prompt
def treatment_choice_json_schema(supplies_json_str, locations_json_str):
    '''
    {"$defs": {"SupplyChoice": {"enum": {{ supplies_json_str }},
       "title": "SupplyChoice",
       "type": "string"},
       "LocationChoice": {"enum": {{ locations_json_str }},
       "title": "LocationChoice",
       "type": "string"}},
     "properties": {"detailed_reasoning": {"title": "Detailed Reasoning",
       "type": "string"},
      "supplies_to_use": {"$ref": "#/$defs/SupplyChoice"},
      "treatment_location": {"$ref": "#/$defs/LocationChoice"}},
     "required": ["detailed_reasoning", "supplies_to_use", "treatment_location"],
     "title": "TreatmentSelection",
     "type": "object"}
    '''


class OutlinesTransformersADM(ActionBasedADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 **kwargs):
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

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        # choices = ["({}) {}".format(chr(i + 65), a.unstructured)
        #            for i, a in enumerate(available_actions)]
        choices = [a.unstructured for a in available_actions]

        scenario_description = scenario_state_description_1(scenario_state)

        prompt = action_selection_prompt(scenario_description, choices)

        dialog = [{'role': 'system', 'content': baseline_system_prompt()},
                  {'role': 'user', 'content': prompt}]
        dialog_text = self.dialog_to_prompt(dialog)

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
            dialog.append({'role': 'user',
                           'content': followup_clarify_tag(
                               scenario_state.characters[selected_character_idx])})

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
