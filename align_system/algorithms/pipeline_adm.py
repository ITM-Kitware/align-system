from align_system.algorithms.abstracts import ActionBasedADM
from align_system.utils import adm_utils
from align_system.utils import logging

log = logging.getLogger(__name__)


class PipelineADM(ActionBasedADM):
    def __init__(self,
                 steps=[]):
        self.steps = steps

    def choose_action(self,
                      scenario_state,
                      available_actions,
                      alignment_target=None):
        # Seed the `choice_evaluation`
        choices = adm_utils.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state)

        choice_evaluation = {}
        for choice, action in zip(choices, available_actions):
            if choice in choice_evaluation:
                raise RuntimeError("Assumption violated that each 'choice' is "
                                   "unique, aborting")
            choice_evaluation[choice] = {"action": action}

        dialogs = [[]]
        for step in self.steps:
            choice_evaluation, dialogs = step.run(scenario_state,
                                                  choice_evaluation,
                                                  dialogs,
                                                  alignment_target)

        import xdev
        xdev.embed()

        chosen_choices = {k: v for k, v in choice_evaluation.items() if v.get('chosen')}

        if len(chosen_choices) != 1:
            raise RuntimeError("Assumption violated that one 'choice' would "
                               "have 'chosen' set to True ({} choices "
                               "chosen)".format(len(chosen_choices)))

        choice, choice_info = chosen_choices.items()[0]

        return choice_info['action'], choice_info
