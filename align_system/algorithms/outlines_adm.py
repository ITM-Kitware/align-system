from enum import StrEnum
import json

import outlines
import jinja2
from pydantic import BaseModel

from align_system.utils import logging
from align_system.algorithms.abstracts import ActionBasedADM

log = logging.getLogger(__name__)


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
def action_choice_json_schema(choices_json_str):
    '''
    {"$defs": {"ActionChoice": {"enum": {{ choices_json_str }},
       "title": "ActionChoice",
       "type": "string"}},
     "properties": {"detailed_resoning": {"title": "Detailed Resoning",
       "type": "string"},
      "action_choice": {"$ref": "#/$defs/ActionChoice"}},
     "required": ["detailed_resoning", "action_choice"],
     "title": "Action",
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
        encoded_dialog = tokenizer.apply_chat_template(dialog)
        prompt = tokenizer.decode(encoded_dialog)

        log.debug("[bold]*DIALOG AS PROMPT*[/bold]",
                  extra={"markup": True})
        log.debug(prompt)

        return prompt

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        choices = [a.unstructured for a in available_actions]

        scenario_description = scenario_state_description_1(scenario_state)

        prompt = action_selection_prompt(scenario_description, choices)

        dialog = [{'role': 'system', 'content': baseline_system_prompt()},
                  {'role': 'user', 'content': prompt}]

        try:
            dialog_text = self.dialog_to_prompt(dialog)
        except jinja2.exceptions.TemplateError:
            # Assume that the tokenizer chat template doesn't accept
            # system messages; combine system message first user
            # message
            system_msg, user_msg, *rest = dialog

            assert user_msg['role'] == 'user'

            updated_content = system_msg['content'] + '\n' + user_msg['content']

            ammended_dialog = [{'role': 'user', 'content': updated_content}, *rest]

            dialog_text = self.dialog_to_prompt(ammended_dialog)

        generator = outlines.generate.json(
            self.model,
            action_choice_json_schema(json.dumps(choices)))

        chosen_action = generator(dialog_text)

        # WIP: not returning a valid Action currently
        return chosen_action
