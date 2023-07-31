import argparse

from swagger_client import ItmTa2EvalApi
from swagger_client.configuration import Configuration
from swagger_client.api_client import ApiClient
from swagger_client import (
    ProbeResponse,
)


class TA3CACIServiceInterface:
    def __init__(self,
                 username='ALIGN-ADM',
                 api_endpoint='http://127.0.0.1:8080',
                 session_type='eval'):
        _config = Configuration()
        _config.host = api_endpoint
        _api_client = ApiClient(configuration=_config)

        self.username = username

        self._client = ItmTa2EvalApi(api_client=_api_client)
        self._client.start_session(adm_name=username,
                                   session_type=session_type)

    def start_scenario(self):
        scenario = self._client.start_scenario(self.username)

        return TA3CACIScenario(self._client, scenario)

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
                            default="eval",
                            help='TA3 API Session Type (default: "eval")')
        parser.add_argument('-e', '--api_endpoint',
                            default="http://127.0.0.1:8080",
                            type=str,
                            help='Restful API endpoint for scenarios / probes '
                                 '(default: "http://127.0.0.1:8080")')

        return parser

    @classmethod
    def cli_parser_description(cls):
        return "Argument parser for TA3CACIServiceInterface"

    @classmethod
    def init_from_parsed_args(cls, parsed_args):
        return cls(**vars(parsed_args))


class TA3CACIScenario:
    def __init__(self, client, scenario):
        self._client = client
        self._scenario = scenario

        self._scenario_complete = False
        self._responded_to_current_probe = True

    def get_alignment_target(self):
        return self._client.get_alignment_target(self._scenario.id)

    def to_dict(self):
        return self._scenario.to_dict()

    def data(self):
        return self._scenario

    def _respond_to_probe_callback(self, probe_id, response_data):
        response_data = {'scenario_id': self._scenario.id,
                         'probe_id': probe_id,
                         **response_data}

        server_response = self._client.respond_to_probe(
            body=ProbeResponse(**response_data))

        self._scenario_complete = server_response.scenario_complete

        self._responded_to_current_probe = True

        return server_response

    def iterate_probes(self):
        while not self._scenario_complete:
            if self._responded_to_current_probe:
                self._responded_to_current_probe = False

                self.current_probe = TA3CACIProbe(
                    self._client.get_probe(self._scenario.id),
                    self._respond_to_probe_callback)

                yield self.current_probe
            else:
                raise RuntimeError(
                    "Must respond to current probe with "
                    "`.respond(response_data)` ")


class TA3CACIProbe:
    def __init__(self, probe_data, response_callback):
        self._probe_data = probe_data
        self._response_callback = response_callback

    def to_dict(self):
        return self._probe_data.to_dict()

    def data(self):
        return self._probe_data

    def respond(self, response_data):
        self._response_callback(self._probe_data.id, response_data)

    def pretty_print_str(self):
        probe = self.to_dict()

        options_string = "\n".join([f"[{o['id']}] {o['value']}"
                                    for o in probe.get('options', [])])

        return f"[{probe['id']}] {probe['prompt']}\n{options_string}"
