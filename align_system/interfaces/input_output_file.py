import argparse
import json

from swagger_client.models import State, Action, Character, Supplies

from align_system.interfaces.abstracts import (
    Interface,
    ActionBasedScenarioInterface)


class InputOutputFileInterface(Interface):
    def __init__(self, input_output_filepath):
        with open(input_output_filepath) as f:
            self._raw_data = json.load(f)

        self.scenario_ids = []
        self.scenarios = {}
        for record in self._raw_data:
            scenario_id = record['input']['scenario_id']

            if scenario_id not in self.scenarios:
                self.scenario_ids.append(scenario_id)
                self.scenarios[scenario_id] = []

            state = State(**record['input']['full_state'])
            # For some reason this initialization from a dictionary
            # doesn't recursively init; need to manually do it
            state.characters = [Character(**c) for c in state.characters]
            state.supplies = [Supplies(**s) for s in state.supplies]

            actions = [Action(**a) for a in record['input']['choices']]
            # TODO: Fix this on the input-output generation side, need
            # to make sure original choices aren't being modified by
            # ADM; for now manually clearing the justification string
            for a in actions:
                a.justification = None

            self.scenarios[scenario_id].append(
                (state, actions))

        self.current_scenario_id = None

    def start_scenario(self):
        if len(self.scenario_ids) > 0:
            self.current_scenario_id, *self.scenario_ids = self.scenario_ids
        else:
            return None

        return InputOutputFileScenario(
            self.current_scenario_id,
            self.scenarios[self.current_scenario_id])

    def get_session_alignment(self, alignment_target_id):
        pass

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
    def __init__(self, scenario_id, scenario_records):
        self.scenario_id = scenario_id
        self.scenario_records = scenario_records

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

    def take_action(self, action):
        if len(self.scenario_records) > 0:
            self.current_state, self.current_actions = self.scenario_records.pop(0)
            return self.current_state
        else:
            self.current_state.scenario_complete = True
            return self.current_state

    def get_state(self):
        return self.current_state
