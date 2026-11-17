"""Host completion for the concrete decoy actions in the CIA scenario."""

import inspect

from align_system.algorithms.cage_action_parameter_completion_adm_component import (
    CAGEActionParameterCompletionADMComponent,
)


class CAGELLMParameterCompletion(CAGEActionParameterCompletionADMComponent):
    def run(self, scenario_state, actions, choices, chosen_choice, dialog=None,
            alignment_target=None, justification=None):
        # Carry the action reasoning into the host-selection follow-up. The
        # legacy pipeline otherwise assigns justification only after this step.
        actions[choices.index(chosen_choice)].justification = justification
        return super().run(scenario_state, actions, choices, chosen_choice,
                           dialog, alignment_target)

    def populate_action_parameters(self, scenario_state, action_to_take, dialog):
        # Includes DecoyApache, DecoySSHD, etc., which the legacy name list omits.
        if 'hostname' in inspect.signature(action_to_take.cage_class).parameters:
            action_to_take, _, _, dialog = self.ensure_hostname_is_populated(
                scenario_state, action_to_take, dialog)
        return action_to_take, dialog
