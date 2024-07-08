import json
import random
import itertools

import outlines
from rich.highlighter import JSONHighlighter
from swagger_client.models import (
    ActionTypeEnum
)

from align_system.utils import logging
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (
    scenario_state_description_1,
    outcomes_system_prompt,
    outcome_prediction_prompt,
    outcome_prediction_json_schema
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()

class OutlinesTransformersRegressionADM(OutlinesTransformersADM):
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


    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_samples=1,
                                shuffle_choices=False,
                                predict_outcomes=False,
                                **kwargs):

        scenario_description = scenario_state_description_1(scenario_state)

        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = [a.unstructured for a in available_actions]

        if len(set(choices)) != len(choices):
            log.warning("Unstructured text for available actions is not "
                        "unique, appending action parameters to choices")

            character_id_to_name = {c.id: c.name for c in scenario_state.characters}
            # Important that the choices stay in the same order as the
            # available actions as we'll use the selected index later to
            # map to the corresponding action
            choices = []
            for a in available_actions:
                if(a.action_type == ActionTypeEnum.APPLY_TREATMENT
                   and a.parameters is not None and len(a.parameters) > 0):
                    choices.append(detailed_unstructured_treatment_action_text(a, character_id_to_name))
                elif(a.action_type == ActionTypeEnum.TAG_CHARACTER
                     and a.parameters is not None and len(a.parameters) > 0):
                    choices.append(detailed_unstructured_tagging_action_text(a, character_id_to_name))
                else:
                    # Not covering every possible case here, may need
                    # to add more dedicated detailed prompts
                    choices.append(a.unstructured)

        kdma_values = alignment_target.kdma_values

        # Loop over samples
        for _ in range(num_samples):
            # Shuffle
            if shuffle_choices:
                shuffled_choices = random.sample(choices, len(choices))
            else:
                shuffled_choices = choices

            # Predict outcome of selecting each choice - optional
            if predict_outcomes:
                outcome_dialogs = []
                outcomes_sys_prompt = outcomes_system_prompt()

                for choice in shuffled_choices:
                    predict_outcome_prompt = outcome_prediction_prompt(scenario_description, choice)
                    outcome_dialogs.append([{'role': 'system', 'content': outcomes_sys_prompt},
                                {'role': 'user', 'content': predict_outcome_prompt}])

                # Need to set the whitespace_pattern to prevent the state
                # machine from looping indefinitely in some cases, see:
                # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
                outcome_generator = outlines.generate.json(
                    self.model,
                    outcome_prediction_json_schema(),
                    whitespace_pattern=r"[ ]?")

                outcome_dialog_texts = [self.dialog_to_prompt(d) for d in
                    itertools.chain(outcome_dialogs)]

                # List of {predicted_outcomes:}, one of each choice in order of shuffled_choices
                predicted_outcomes = outcome_generator(outcome_dialog_texts)

            else:
                predicted_outcomes = None

            import IPython
            IPython.embed()
            # Predict KDMA values for each choice
    