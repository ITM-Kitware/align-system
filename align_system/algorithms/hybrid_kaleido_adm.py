from align_system.utils import logging
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.prompt_engineering.outlines_prompts import (
    baseline_system_prompt,
)

log = logging.getLogger(__name__)


class HybridKaleidoADM(ActionBasedADM):
    def __init__(self, kaleido_adm, outlines_adm):
        self.kaleido_adm = kaleido_adm

        self.outlines_adm = outlines_adm

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        action_to_take, choice_info = self.kaleido_adm.choose_action(
            scenario_state, available_actions, alignment_target, **kwargs)

        # Build out initial dialog for the outlines ADM in order to
        # fill in remaining parameters
        prompt, _ = self.outlines_adm._state_to_top_level_prompt(
            scenario_state, available_actions)

        dialog = [{'role': 'system', 'content': baseline_system_prompt()}]
        dialog.append({'role': 'user', 'content': prompt})

        self.outlines_adm.populate_action_parameters(
            scenario_state,
            action_to_take,
            dialog)

        return action_to_take, choice_info
