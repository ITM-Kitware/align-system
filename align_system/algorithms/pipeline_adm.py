from collections.abc import Iterable
import inspect
from inspect import Parameter

from align_system.algorithms.abstracts import ActionBasedADM, ADMComponent
from align_system.utils import adm_utils
from align_system.utils import logging

log = logging.getLogger(__name__)


class PipelineADM(ActionBasedADM):
    def __init__(self, steps: dict[str, ADMComponent]):
        '''
        Expecting `steps` to be a dictionary of {"<step number>":
        <ADMComponent instance>}.  E.g. {"0":
        <MyADMComponent1>}. `steps` is a dictionary instead of a
        simple list due to limitations with Hydra configurations and
        not being able to set values in a list from the defaults list
        '''
        self.steps = steps

    def _steps_iterator(self):
        sorted_step_nums = sorted([int(k) for k in self.steps.keys()])

        for s in sorted_step_nums:
            yield self.steps[str(s)]

    def choose_action(self,
                      scenario_state,
                      available_actions,
                      alignment_target=None):
        # Seed the `choice_evaluation`
        choices = adm_utils.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state)

        working_output = {'scenario_state': scenario_state,
                          'choices': choices,
                          'actions': available_actions,
                          'alignment_target': alignment_target}
        for step in self._steps_iterator():
            step_returns = step.run_returns()

            # TODO: May need more robust checking around parameter types here
            args = []
            kwargs = {}
            for name, param in inspect.signature(step.run).parameters.items():
                if name in working_output:
                    if param.kind == Parameter.POSITIONAL_ONLY:
                        # Rare case, usually parameters are of kind
                        # POSITIONAL_OR_KEYWORD
                        args.append(working_output[name])
                    else:
                        kwargs[name] = working_output[name]
                elif param.default != inspect._empty:
                    pass  # Don't need to add to the arg/kwarg list
                else:
                    raise RuntimeError(f"Don't have expected parameter "
                                       f"('{name}') in working_output")

            log.debug(f"Args for next step: {args}")
            log.debug(f"Kwargs for next step: {kwargs}")
            run_output = step.run(*args, **kwargs)

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

        return working_output['chosen_action']
