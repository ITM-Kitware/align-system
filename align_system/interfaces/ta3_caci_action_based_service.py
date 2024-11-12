import argparse
from uuid import uuid4

import swagger_client
from swagger_client.configuration import Configuration
from swagger_client.api_client import ApiClient
from swagger_client.models import Action

from align_system.utils import logging
from align_system.interfaces.abstracts import (
    Interface,
    ActionBasedScenarioInterface)


log = logging.getLogger(__name__)


class TA3CACIActionBasedServiceInterface(Interface):
    def __init__(self,
                 username='ALIGN-ADM',
                 api_endpoint='http://127.0.0.1:8080',
                 session_type='eval',
                 scenario_ids=[],
                 training_session=None):
        self.api_endpoint = api_endpoint
        # Append a UUID onto the end of our username, as the TA3
        # server doesn't allow multiple concurrent sessions for the
        # same ADM (by name)
        self.username = "{}__{}".format(username, uuid4())
        self.scenario_ids = scenario_ids
        if len(self.scenario_ids) > 0:
            self.scenarios_specified = True
        else:
            self.scenarios_specified = False

        self.training_session = training_session

        config = Configuration()
        config.host = self.api_endpoint
        api_client = ApiClient(configuration=config)
        self.connection = swagger_client.ItmTa2EvalApi(api_client=api_client)

        start_session_params = {'adm_name': self.username,
                                'session_type':  session_type}

        if self.training_session is not None:
            if self.training_session not in {'full', 'solo'}:
                raise RuntimeError("Expecting `training_session` to be "
                                   "either 'full' or 'solo'")

            start_session_params['kdma_training'] = self.training_session

        self.session_id = self.connection.start_session(
            **start_session_params)

    def start_scenario(self):
        log.info(f"*ADM Name*: {self.username}")

        scenario_request_params = {'session_id': self.session_id}
        if len(self.scenario_ids) > 0:
            scenario_id = self.scenario_ids.pop(0)
            scenario_request_params['scenario_id'] = scenario_id
        elif self.scenarios_specified:
            # Have run through all specified scenarios
            return None

        scenario = self.connection.start_scenario(
            **scenario_request_params)

        return TA3CACIActionBasedScenario(
            self.connection, self.session_id, scenario)

    def get_session_alignment(self, alignment_target):
        if self.training_session == 'full':
            # 'solo' training sessions are not able to retrieve an
            # alignment score
            return self.connection.get_session_alignment(
                self.session_id, alignment_target.id)
        else:
            return None

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
        parser.add_argument('-S', '--scenario-id',
                            dest='scenario_ids',
                            required=False,
                            default=[],
                            nargs='*',
                            help='Specific scenario to run (multiples allowed)')

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

    def id(self):
        return self.scenario.id

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

    def _take_or_intend_action(self, action, take_or_intend):
        # Convert to proper 'Action' object prior to submission
        if isinstance(action, dict):
            action = Action(**action)

        updated_state = take_or_intend(
            session_id=self.session_id,
            body=action)

        return updated_state

    def intend_action(self, action):
        return self._take_or_intend_action(
            action, self.connection.intend_action
        )

    def take_action(self, action):
        return self._take_or_intend_action(
            action, self.connection.take_action
        )

    def get_state(self):
        return self.connection.get_scenario_state(
            session_id=self.session_id, scenario_id=self.scenario.id)
