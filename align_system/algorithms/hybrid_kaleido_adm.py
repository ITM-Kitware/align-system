from swagger_client.models import ActionTypeEnum

from align_system.utils import logging
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.algorithms.llama_2_single_kdma_adm import Llama2SingleKDMAADM
from align_system.algorithms.kaleido_adm import KaleidoADM

log = logging.getLogger(__name__)


class HybridKaleidoADM(ActionBasedADM):
    def __init__(self, **kwargs):
        self.kaleido_adm = KaleidoADM(**kwargs.get('kaleido_init_kwargs', {}))

        self.llm_algorithm = Llama2SingleKDMAADM(**kwargs.get('llm_init_kwargs', {}))
        self.llm_algorithm.load_model()

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        action_to_take = self.kaleido_adm.choose_action(
            scenario_state, available_actions, alignment_target, **kwargs)

        if action_to_take.action_type == ActionTypeEnum.APPLY_TREATMENT:
            # If the additional required fields are already populated
            # for the action, don't need ask the LLM again
            if (action_to_take.parameters is None
                or not {'treatment', 'location'}.issubset(
                    action_to_take.parameters.keys())):
                action_to_take = self.llm_algorithm.populate_treatment_parameters(
                    scenario_state, action_to_take, alignment_target, **kwargs)
        elif action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            # If the additional required fields are already populated
            # for the action, don't need ask the LLM again
            if (action_to_take.character_id is None
                or action_to_take.parameters is None
                or not {'category'}.issubset(
                    action_to_take.parameters.keys())):
                action_to_take = self.llm_algorithm.populate_tagging_parameters(
                    scenario_state, action_to_take, alignment_target, **kwargs)
        elif action_to_take.action_type in {ActionTypeEnum.CHECK_ALL_VITALS,
                                            ActionTypeEnum.CHECK_PULSE,
                                            ActionTypeEnum.CHECK_RESPIRATION,
                                            ActionTypeEnum.MOVE_TO_EVAC,
                                            ActionTypeEnum.CHECK_BLOOD_OXYGEN}:
            # These actions require a `character_id`
            if action_to_take.character_id is None:
                action_to_take = self.llm_algorithm.generic_populate_character_id(
                    scenario_state, action_to_take, alignment_target, **kwargs)

        return action_to_take
