import json
from collections import defaultdict
from rich.highlighter import JSONHighlighter
from swagger_client.models import ActionTypeEnum, CharacterTagEnum

from align_system.algorithms.abstracts import ADMComponent
from align_system.algorithms.outlines_baseline_adm_component import OutlinesBaselineADMComponent
from align_system.data_models.dialog import DialogElement
from align_system.prompt_engineering.outlines_prompts import character_choice_json_schema, tag_choice_json_schema
from align_system.prompt_engineering.ow_prompts import FollowupClarifyCharacterPrompt, FollowupClarifyTagPrompt
from align_system.utils import call_with_coerced_args, logging, get_swagger_class_enum_values

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class OWFormatChoicesADMComponent(ADMComponent):
    def run_returns(self):
        return ('choices', 'choice_to_action_mapping')

    def run(self, scenario_state, actions):
        choice_to_action_mapping = defaultdict(list)
        choices = []  # Not a set to preserve action order

        non_character_action_types = [
            ActionTypeEnum.END_SCENE
        ]

        character_to_choice = {
            c.id: f"{c.name}: {c.unstructured}"
            for c in scenario_state.characters
        }

        def _add_choice(choice, action):
            """Add choice to list of choices (if needed), add action to mapping."""
            if choice not in choices:
                choices.append(choice)
            choice_to_action_mapping[choice].append(action)

        # Add each character/non-character action to list of choices and
        # construct a choice to action mapping for later use
        for a in actions:
            if a.character_id is not None:
                _add_choice(character_to_choice[a.character_id], a)
            elif a.action_type in non_character_action_types:
                _add_choice(a.unstructured, a)
            else:  # No character specified, add all characters
                for c in scenario_state.characters:
                    _add_choice(character_to_choice[c.id], a)

        return choices, choice_to_action_mapping


class OWChoiceToActionADMComponent(OutlinesBaselineADMComponent):
    def run_returns(self):
        return ('chosen_action', 'choice_to_action_dialog')

    def run(
        self,
        scenario_state,
        chosen_choice,
        choice_to_action_mapping,
        justification=None,
    ):
        if chosen_choice not in choice_to_action_mapping:
            raise ValueError(f"Choice ({chosen_choice}) not found in choice_to_action_mapping")

        choice_to_action_dialog = None
        possible_actions = choice_to_action_mapping[chosen_choice]

        if len(possible_actions) == 0:
            raise ValueError(f"Choice ({chosen_choice}) has no possible actions")
        elif len(possible_actions) == 1:  # Single action, choose that
            chosen_action = possible_actions[0]
        else:
            choices = [a.unstructured for a in possible_actions]
            chosen_choice, justification, choice_to_action_dialog = super().run(scenario_state, choices)

            chosen_action = possible_actions[choices.index(chosen_choice)]

        if (hasattr(chosen_action, 'justification')
                and chosen_action.justification is None
                and justification is not None):
            if isinstance(chosen_action, tuple) and hasattr(chosen_action, "_replace"):
                chosen_action = chosen_action._replace(justification=justification)
            else:
                chosen_action.justification = justification

        return chosen_action, choice_to_action_dialog


class OWActionParameterCompletionADMComponent(ADMComponent):
    def __init__(
        self,
        structured_inference_engine,
        scenario_description_template,
        system_prompt=None,
    ):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.system_prompt = system_prompt

        self.followup_character_prompt = FollowupClarifyCharacterPrompt()
        self.followup_tag_prompt = FollowupClarifyTagPrompt()

    def run_returns(self):
        return ('chosen_action', 'action_parameter_completion_dialog')

    def run(
        self,
        scenario_state,
        chosen_action,
    ):
        action_parameter_completion_dialog = {}

        # Action requires a character ID
        if chosen_action.action_type in {'TREAT_PATIENT',
                                         ActionTypeEnum.MOVE_TO_EVAC,
                                         ActionTypeEnum.TAG_CHARACTER}:
            if chosen_action.character_id is None:
                dialog = []
                if self.system_prompt is not None:
                    dialog.append(DialogElement(role='system', content=self.system_prompt()))

                scenario_description = call_with_coerced_args(
                    self.scenario_description_template,
                    {'scenario_state': scenario_state})

                dialog.append(
                    DialogElement(
                        role='user',
                        content=self.followup_character_prompt(scenario_description, chosen_action)
                    )
                )

                dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
                character_names = [c.name for c in scenario_state.characters]
                selected_character = self.structured_inference_engine.run_inference(
                    dialog_prompt,
                    character_choice_json_schema(json.dumps(character_names)),
                )
                selected_character_idx = character_names.index(selected_character['character_choice'])

                chosen_action.character_id = scenario_state.characters[selected_character_idx].id

                justification = selected_character["brief_reasoning"]
                if isinstance(chosen_action, tuple) and hasattr(chosen_action, "_replace"):
                    chosen_action = chosen_action._replace(justification=justification)
                else:
                    chosen_action.justification = justification

                action_parameter_completion_dialog["character_id"] = dialog

        # Tagging requires tag type
        if chosen_action.action_type == ActionTypeEnum.TAG_CHARACTER:
            if chosen_action.parameters is None:
                chosen_action.parameters = {}

            if 'category' not in chosen_action.parameters:
                dialog = []
                if self.system_prompt is not None:
                    dialog.append(DialogElement(role='system', content=self.system_prompt()))

                chosen_character = None
                for c in scenario_state.characters:
                    if c.id == chosen_action.character_id:
                        chosen_character = c
                        break

                dialog.append(
                    DialogElement(role='user', content=self.followup_tag_prompt(chosen_character))
                )

                dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
                valid_tags = get_swagger_class_enum_values(CharacterTagEnum)
                selected_tag = self.structured_inference_engine.run_inference(
                    dialog_prompt,
                    tag_choice_json_schema(json.dumps(valid_tags))
                )

                chosen_action.parameters['category'] = selected_tag["triage_tag"]

                justification = selected_tag["detailed_reasoning"]
                if isinstance(chosen_action, tuple) and hasattr(chosen_action, "_replace"):
                    chosen_action = chosen_action._replace(justification=justification)
                else:
                    chosen_action.justification = justification

                action_parameter_completion_dialog["tag"] = dialog

        return chosen_action, action_parameter_completion_dialog


class OWTaggingAdjustmentADMComponent(ADMComponent):
    def run_returns(self):
        return ('chosen_action')

    def run(
        self,
        scenario_state,
        chosen_action,
        choices,
        choice_to_action_mapping,
        attribute_prediction_scores,
        p_choices,
    ):
        if chosen_action.action_type == ActionTypeEnum.TAG_CHARACTER:
            # TODO: Affect tagging behavior
            pass

        return chosen_action
