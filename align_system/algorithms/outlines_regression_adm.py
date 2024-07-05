import json
import random

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
    outcomes_prediction_prompt
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

        for _ in range(num_samples):
            if shuffle_choices:
                shuffled_choices = random.sample(choices, len(choices))
            else:
                shuffled_choices = choices

            if predict_outcomes:
                outcomes_sys_prompt = outcomes_system_prompt()
                predict_outcomes_prompt = outcomes_prediction_prompt(scenario_description, shuffled_choices)
                dialog = [{'role': 'system', 'content': outcomes_sys_prompt},
                          {'role': 'user', 'content': predict_outcomes_prompt}]

        import IPython
        IPython.embed()

    