import requests

from align_system.interfaces.abstracts import (
    Interface,
    ScenarioInterfaceWithAlignment,
    ProbeInterfaceWithAlignment)


class TA1SoartechServiceInterface(Interface):
    def __init__(self,
                 api_endpoint='http://127.0.0.1:8084',
                 scenarios=['kickoff-demo-scenario-1'],
                 alignment_targets=['kdma-alignment-target-1']):
        self.api_endpoint = api_endpoint

        session = requests.post(
            f"{self.api_endpoint}/api/v1/new_session")
        self.session_id = session.json()  # Should be single string response

        if len(scenarios) != len(alignment_targets):
            raise RuntimeError(
                f"Length of scenarios ({len(scenarios)}) doesn't match "
                f"length of alignment targets ({len(alignment_targets)})")

        self.scenarios = scenarios
        self.alignment_targets = alignment_targets

        self.remaining_scenarios = zip(self.scenarios, self.alignment_targets)

    def start_scenario(self):
        try:
            scenario_id, alignment_target_id = next(self.remaining_scenarios)
        except StopIteration:
            return None
        else:
            return TA1SoartechScenario(
                self.api_endpoint,
                self.session_id,
                scenario_id, alignment_target_id)


class TA1SoartechScenario(ScenarioInterfaceWithAlignment):
    def __init__(self, api_endpoint, session_id, scenario, alignment_target):
        self.api_endpoint = api_endpoint
        self.session_id = session_id

        self.scenario_id = scenario
        self.alignment_target_id = alignment_target

        self._scenario = requests.get(
            f"{self.api_endpoint}/api/v1/scenario/{self.scenario_id}").json()

        self._alignment_target = None

    def get_alignment_target(self):
        if self._alignment_target is None:
            self._alignment_target = requests.get(
                f"{self.api_endpoint}/api/v1/alignment_target/{self.alignment_target_id}").json()  # noqa E501

        return self._alignment_target

    def to_dict(self):
        return self._scenario

    def data(self):
        return self._scenario

    def _respond_to_probe_callback(self, probe_id, response_data):
        return requests.post(
            f"{self.api_endpoint}/api/v1/response",
            json={'session_id': self.session_id,
                  'response': {'scenario_id': self.scenario_id,
                               'probe_id': probe_id,
                               **response_data}}).json()

    def _get_probe_alignment_score_callback(self, probe_id):
        return requests.get(
            f"{self.api_endpoint}/api/v1/alignment/probe",
            params={'session_id': self.session_id,
                    'target_id': self.alignment_target_id,
                    'scenario_id':  self.scenario_id,
                    'probe_id': probe_id}).json()

    def iterate_probes(self):
        for probe in self._scenario.get('probes', ()):
            probe_obj = TA1SoartechProbe(
                probe,
                self._respond_to_probe_callback,
                self._get_probe_alignment_score_callback)

            yield probe_obj

    def get_alignment_results(self):
        return requests.get(
            f"{self.api_endpoint}/api/v1/alignment/session",
            params={'session_id': self.session_id,
                    'target_id': self.alignment_target_id}).json()


class TA1SoartechProbe(ProbeInterfaceWithAlignment):
    def __init__(self,
                 probe_data,
                 response_callback,
                 probe_alignment_score_callback):
        self._probe_data = probe_data
        self._response_callback = response_callback
        self._probe_alignment_score_callback = probe_alignment_score_callback

    def to_dict(self):
        return self._probe_data

    def data(self):
        return self._probe_data

    def respond(self, response_data):
        self._response_callback(self._probe_data['id'], response_data)

    def get_alignment_results(self):
        return self._probe_alignment_score_callback(self._probe_data['id'])

    def pretty_print_str(self):
        probe = self.to_dict()

        options_string = "\n".join([f"[{o['id']}] {o['value']}"
                                    for o in probe.get('options', [])])

        return f"[{probe['id']}] {probe['prompt']}\n{options_string}"
