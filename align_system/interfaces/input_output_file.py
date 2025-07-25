import argparse
import json
import math

from align_system.interfaces.abstracts import (
    Interface,
    ActionBasedScenarioInterface)


class InputOutputFileInterface(Interface):
    def __init__(self, input_output_filepath, state_hydration_domain=None):
        with open(input_output_filepath) as f:
            self._raw_data = json.load(f)

        # TODO: Replace this logic with a callable to be instantiated via Hydra
        if state_hydration_domain is None or state_hydration_domain == "p1":
            from align_system.utils.hydrate_state import hydrate_scenario_state
            state_hydration_fn = hydrate_scenario_state
        elif state_hydration_domain == "p2triage":
            from align_system.utils.hydrate_state import p2triage_hydrate_scenario_state
            state_hydration_fn = p2triage_hydrate_scenario_state
        elif state_hydration_domain == "minimal":
            from align_system.utils.hydrate_state import minimal_hydrate_scenario_state
            state_hydration_fn = minimal_hydrate_scenario_state
        else:
            raise RuntimeError(f"Unknown state_hydration_domain: {state_hydration_domain}")

        self.scenario_ids = []
        self.scenarios = {}
        for record in self._raw_data:
            scenario_id = record['input']['scenario_id']

            if scenario_id not in self.scenarios:
                self.scenario_ids.append(scenario_id)
                self.scenarios[scenario_id] = []

            self.scenarios[scenario_id].append(
                state_hydration_fn(record['input']))

        self.current_scenario_id = None

        # Items should be tuple of (scenario_id, possible_actions, action_taken)
        self.action_history = []

    def start_scenario(self):
        if len(self.scenario_ids) > 0:
            self.current_scenario_id, *self.scenario_ids = self.scenario_ids
        else:
            return None

        return InputOutputFileScenario(
            self.current_scenario_id,
            self.scenarios[self.current_scenario_id],
            self._action_taken_callback)

    def _action_taken_callback(self, scenario_id, available_actions, taken_action):
        self.action_history.append((scenario_id, available_actions, taken_action))

    def get_session_alignment(self, alignment_target):
        targ = {kv['kdma']: kv['value'] for kv in alignment_target.kdma_values}

        def _compute_alignment(action):
            if action.kdma_association is None:
                return math.inf

            dists = {}
            for k, v in targ.items():
                # Fail if the expected KDMA is not in the associations
                action_value = action.kdma_association[k]

                dists[k] = abs(action_value - v)

            return sum(dists.values())

        corrects = {}
        corrects_valid_only = {}
        for scenario_id, possible_action, action_taken in self.action_history:
            action_dists = [_compute_alignment(a) for a in possible_action]
            taken_action_dist = _compute_alignment(action_taken)

            is_correct = 1 if min(action_dists) == taken_action_dist else 0
            corrects.setdefault(scenario_id, []).append(is_correct)

            # "Valid" here means that there was more than one action
            # available with a unique KDMA value.  I.e. if there are
            # only two choices each with a KDMA value of 0.5, no
            # accuracy figure is recorded
            valid_action_dists = set(d for d in action_dists if d != math.inf)
            if len(valid_action_dists) > 1:
                is_correct_wrt_valid = 1 if min(valid_action_dists) == taken_action_dist else 0
                corrects_valid_only.setdefault(scenario_id, []).append(is_correct_wrt_valid)
            else:
                continue

        output_measures = {}
        for scenario_id in corrects.keys():
            output_measures.setdefault(scenario_id, {})

            output_measures[scenario_id]['num_correct'] =\
                {'value': sum(corrects[scenario_id]),
                 'description': "Numer of probes where the ADM chose "
                 "the action with the KDMA values closest to the "
                 "alignment target"}

            output_measures[scenario_id]['num_probes'] =\
                {'value': len(corrects[scenario_id]),
                 'description': "Total number of probes"}

            output_measures[scenario_id]['accuracy'] =\
                {'value': sum(corrects[scenario_id]) / len(corrects[scenario_id]),
                 'description': "Numer of probes where the ADM chose "
                 "the action with the KDMA values closest to the "
                 "alignment target over total number of probes"}

            output_measures[scenario_id]['num_correct_valid_only'] =\
                {'value': sum(corrects_valid_only[scenario_id]),
                 'description': "Number of probes where the ADM chose "
                 "the action with the KDMA values closest to the "
                 "alignment target ignoring probes with only a "
                 "single option with respect to unique KDMA values "
                 "(i.e. if there are only two choices each with a "
                 "KDMA value of 0.5 the probe is ignored)"}

            output_measures[scenario_id]['num_probes_valid_only'] =\
                {'value': len(corrects_valid_only[scenario_id]),
                 'description': "Total number of probes"
                 "ignoring probes with only a "
                 "single option with respect to unique KDMA values "
                 "(i.e. if there are only two choices each with a "
                 "KDMA value of 0.5 the probe is ignored)"}

            output_measures[scenario_id]['accuracy_valid_only'] =\
                {'value': sum(corrects_valid_only[scenario_id]) / len(corrects_valid_only[scenario_id]),
                 'description': "Number of probes where the ADM chose "
                 "the action with the KDMA values closest to the "
                 "alignment target over total number of probes "
                 "ignoring probes with only a single option with "
                 "respect to unique KDMA values (i.e. if there are "
                 "only two choices each with a KDMA value of 0.5 "
                 "the probe is ignored)"}

        return {'alignment_target': alignment_target.id,
                'measures': output_measures}


    @classmethod
    def cli_parser(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                description=cls.cli_parser_description())

        parser.add_argument('-i', '--input-output-filepath',
                            type=str,
                            required=True,
                            help='Path to input-output JSON file')

        return parser

    @classmethod
    def cli_parser_description(cls):
        return "Interface with an input-output JSON file"

    @classmethod
    def init_from_parsed_args(cls, parsed_args):
        return cls(**vars(parsed_args))


class InputOutputFileScenario(ActionBasedScenarioInterface):
    def __init__(self, scenario_id, scenario_records, action_taken_callback):
        self.scenario_id = scenario_id
        self.scenario_records = scenario_records

        self.action_taken_callback = action_taken_callback

        self.current_state, self.current_actions = self.scenario_records.pop(0)

    def id(self):
        return self.scenario_id

    def get_alignment_target(self):
        pass

    def to_dict(self):
        return self.current_state.__dict__

    def data(self):
        return self.current_state

    def get_available_actions(self):
        return self.current_actions

    def _take_or_intend_action(self, action):
        self.action_taken_callback(self.scenario_id, self.current_actions, action)

        if len(self.scenario_records) > 0:
            self.current_state, self.current_actions = self.scenario_records.pop(0)
            return self.current_state
        else:
            if isinstance(self.current_state, tuple) and hasattr(self.current_state, "_replace"):
                self.current_state = self.current_state._replace(scenario_complete=True)
            else:
                self.current_state.scenario_complete = True
            return self.current_state

    def intend_action(self, action):
        return self._take_or_intend_action(action)

    def take_action(self, action):
        return self._take_or_intend_action(action)


    def get_state(self):
        return self.current_state
