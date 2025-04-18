from collections.abc import Iterable

from align_system.algorithms.abstracts import ActionBasedADM, ADMComponent
from align_system.utils import logging, call_with_coerced_args

log = logging.getLogger(__name__)


class PipelineADM(ActionBasedADM):
    def __init__(self, steps: list[ADMComponent]):
        self.steps = steps

    def choose_action(self,
                      scenario_state,
                      available_actions,
                      alignment_target=None,
                      **kwargs):

        working_output = {'scenario_state': scenario_state,
                          'actions': available_actions,
                          'alignment_target': alignment_target,
                          **kwargs}

        for step in self.steps:
            step_returns = step.run_returns()

            run_output = call_with_coerced_args(step.run, working_output)

            if isinstance(step_returns, str):
                if step_returns in working_output:
                    log.debug(f"Updating '{step_returns}'")

                working_output[step_returns] = run_output
            elif isinstance(step_returns, Iterable):
                if len(step_returns) != len(run_output):
                    raise RuntimeError(
                        "step_returns and run_output aren't equal length")

                for r, o in zip(step_returns, run_output):
                    if r in working_output:
                        log.debug(f"Updating '{r}'")

                    working_output[r] = o
            else:
                raise TypeError("Don't know how to deal with step returns")

        return working_output['chosen_action']
