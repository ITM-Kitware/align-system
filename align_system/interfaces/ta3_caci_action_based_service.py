import argparse

import swagger_client
from swagger_client.configuration import Configuration
from swagger_client.api_client import ApiClient
from swagger_client.models import Action

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
        self.training_session = training_session

        config = Configuration()
        config.host = self.api_endpoint
        api_client = ApiClient(configuration=config)
        self.connection = swagger_client.ItmTa2EvalApi(api_client=api_client)

        start_session_params = {'adm_name': username,
                                'session_type':  session_type}

        if self.training_session:
            start_session_params['kdma_training'] = True

        self.session_id = self.connection.start_session(
            **start_session_params)

    def start_scenario(self):
        scenario_request_params = {'session_id': self.session_id}
        if self.scenario_id is not None:
            scenario_request_params['scenario_id'] = self.scenario_id

        scenario = self.connection.start_scenario(
            **scenario_request_params)

        return TA3CACIActionBasedScenario(
            self.connection, self.session_id, scenario)

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
    def __init__(self, connection, session_id, scenario):
        self.connection = connection
        self.session_id = session_id

        self.scenario = scenario

    def get_alignment_target(self):
        return self.connection.get_alignment_target(
            self.session_id, self.scenario.id)

    def to_dict(self):
        return self.scenario.__dict__

    def data(self):
        return self.scenario

    def get_available_actions(self):
        return self.connection.get_available_actions(
            session_id=self.session_id, scenario_id=self.scenario.id)

    def take_action(self, action):
        # Convert to proper 'Action' object prior to submission
        if isinstance(action, dict):
            action = Action(**action)

        updated_state = self.connection.take_action(
            session_id=self.session_id,
            body=action)

        return updated_state

    def get_state(self):
        return self.connection.get_scenario_state(
            session_id=self.session_id, scenario_id=self.scenario.id)
