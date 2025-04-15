from collections.abc import Iterable

from align_system.algorithms.abstracts import ActionBasedADM, ADMComponent
from align_system.utils import adm_utils, logging, call_with_coerced_args

log = logging.getLogger(__name__)


class PipelineADM(ActionBasedADM):
    def __init__(self, steps: list[ADMComponent]):
        self.steps = steps

    def choose_action(self,
                      scenario_state,
                      available_actions,
                      alignment_target=None):
        choices = adm_utils.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state)

        working_output = {'scenario_state': scenario_state,
                          'choices': choices,
                          'actions': available_actions,
                          'alignment_target': alignment_target}
        for step in self.steps:
            step_returns = step.run_returns()

            run_output = call_with_coerced_args(step.run, working_output)

            if isinstance(step_returns, str):
                if step_returns in working_output:
                    log.debug(f"Updating '{step_returns}'")

                working_output[step_returns] = run_output
            elif isinstance(step_returns, Iterable):
                for r, o in zip(step_returns, run_output):
                    if r in working_output:
                        log.debug(f"Updating '{r}'")

                    working_output[r] = o
            else:
                raise TypeError("Don't know how to deal with step returns")

        if 'chosen_action' not in working_output:
            if 'chosen_choice' in working_output:
                chosen_choice_idx = working_output['choices'].index(
                    working_output['chosen_choice'])
                working_output['chosen_action'] = working_output['actions'][chosen_choice_idx]
            else:
                raise RuntimeError("Expecting a 'chosen_action' or "
                                   "'chosen_choice' at the end of pipeline run")

        if (hasattr(working_output['chosen_action'], 'justification')
                and working_output['chosen_action'].justification is None
                and 'justification' in working_output):
            working_output['chosen_action'].justification = working_output['justification']

        return working_output['chosen_action']
