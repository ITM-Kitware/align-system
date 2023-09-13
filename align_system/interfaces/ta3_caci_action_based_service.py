import argparse

import requests

from align_system.interfaces.abstracts import (
    Interface,
    ActionBasedScenarioInterface)


class TA3CACIActionBasedServiceInterface(Interface):
    def __init__(self,
                 username='ALIGN-ADM',
                 api_endpoint='http://127.0.0.1:8080',
                 session_type='eval',
                 scenario_id=None,
                 training_session=False):
        self.api_endpoint = api_endpoint
        self.username = username
        self.scenario_id = scenario_id
        self.training_session = False

        start_session_params = {'adm_name': username,
                                'session_type':  session_type}

        if self.training_session:
            start_session_params['kdma_training'] = True

        session = requests.get(
            f"{self.api_endpoint}/ta2/startSession",
            params=start_session_params)

        self.session_id = session.json()  # Should be single string response

    def start_scenario(self):
        scenario_request_params = {'session_id': self.session_id}
        if self.scenario_id is not None:
            scenario_request_params['scenario_id'] = self.scenario_id

        scenario = requests.get(
            f"{self.api_endpoint}/ta2/scenario",
            params=scenario_request_params)

        return TA3CACIActionBasedScenario(
            self.api_endpoint, self.session_id, scenario.json())

    @classmethod
    def cli_parser(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                description=cls.cli_parser_description())

        parser.add_argument('-u', '--username',
                            type=str,
                            default='ALIGN-ADM',
                            help='ADM Username (provided to TA3 API server, '
                                 'default: "ALIGN-ADM")')
        parser.add_argument('-s', '--session-type',
                            type=str,
                            default="test",
                            help='TA3 API Session Type (default: "eval")')
        parser.add_argument('-e', '--api_endpoint',
                            default="http://127.0.0.1:8080",
                            type=str,
                            help='Restful API endpoint for scenarios / probes '
                                 '(default: "http://127.0.0.1:8080")')
        parser.add_argument('--training-session',
                            action='store_true',
                            default=False,
                            help='Return training related information from '
                                 'API requests')

        return parser

    @classmethod
    def cli_parser_description(cls):
        return "Interface with CACI's TA3 web-based service"

    @classmethod
    def init_from_parsed_args(cls, parsed_args):
        return cls(**vars(parsed_args))


class TA3CACIActionBasedScenario(ActionBasedScenarioInterface):
    def __init__(self, api_endpoint, session_id, scenario):
        self.api_endpoint = api_endpoint
        self.session_id = session_id

        self._scenario = scenario
        self.scenario_id = scenario['id']

    def get_alignment_target(self):
        alignment_target = requests.get(
            f"{self.api_endpoint}/ta2/getAlignmentTarget",
            params={'session_id': self.session_id,
                    'scenario_id': self.scenario_id})

        return alignment_target.json()

    def to_dict(self):
        return self._scenario

    def data(self):
        return self._scenario

    def get_available_actions(self):
        available_actions = requests.get(
            f"{self.api_endpoint}/ta2/{self.scenario_id}/getAvailableActions",
            params={'session_id': self.session_id})

        return available_actions.json()

    def take_action(self, action_data):
        updated_state = requests.post(
            f"{self.api_endpoint}/ta2/takeAction",
            params={'session_id': self.session_id},
            json=action_data)

        if updated_state.status_code == 400:
            raise RuntimeError("Bad client request, action_data is either in "
                               "the wrong format, or doesn't include the "
                               "required fields")
        elif updated_state.status_code == 500:
            raise RuntimeError("TA3 internal server error!")
        elif updated_state.status_code != 200:
            raise RuntimeError("'takeAction' didn't succeed (returned status "
                               "code: {})".format(updated_state.status_code))

        return updated_state.json()

    def get_state(self):
        state = requests.get(
            f"{self.api_endpoint}/ta2/{self.scenario_id}/getState",
            params={'session_id': self.session_id})

        return state.json()
