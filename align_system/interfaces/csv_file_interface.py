import json
import pandas as pd

from align_system.interfaces.abstracts import (
    Interface,
    ScenarioInterface,
    ProbeInterface)


class CSVDatasetInterface(Interface):
    def __init__(self, csv_file):
        # {scenario_id: [list of probes]}
        # list_of_scenarios
        df = pd.read_csv(csv_file)
        samples = []
        for probe_id, probe_df in df.groupby('probe_id'):
            sample = None
            for i, row in probe_df.iterrows():
                if sample is None:
                    sample = {
                        'scenario_id': row['scenario_id'],
                        'probe_id': row['probe_id'],
                        'scenario': row['scenario'],
                        'state': row['state'] if not pd.isna(row['state']) else None,
                        'probe': row['probe']
                    }
                    sample['choices'] = []
                    samples.append(sample)

                sample['choices'].append({
                    'text': row['answer'],
                    'kdmas': {
                        kdma: row[kdma]
                        for kdma in ['basic_knowledge', 'time_pressure', 'risk_aversion', 'fairness', 'protocol_focus', 'utilitarianism']
                        if not pd.isna(row[kdma])
                    }
                })

        inputs = []
        labels = []
        for sample in samples:
            inputs.append({
                key: value
                for key, value in sample.items() if key != 'choices'
            })
            inputs[-1]['choices'] = [
                choice['text']
                for choice in sample['choices']
            ]
            labels.append([
                choice['kdmas']
                for choice in sample['choices']
            ])
            # delete kdmas from choices
            for choice in sample['choices']:
                del choice['kdmas']


        self.scenarios = {}
        for input_, label in zip(inputs, labels):
            scenario_id = input_['scenario_id']
            if scenario_id not in self.scenarios:
                self.scenarios[scenario_id] = []
            # append tuple
            self.scenarios[scenario_id].append((input_, label))

        self.scenario_ids = list(self.scenarios.keys())
        self.scenario_id_iterator = iter(self.scenario_ids)

    def start_scenario(self):
        try:
            scenario_id = next(self.scenario_iterator)
        except StopIteration:
            return None
        return CSVDatasetScenario(self.scenarios[scenario_id])


class CSVDatasetScenario(ScenarioInterface):

    def __init__(self, list_of_probes):
        self.list_of_probes = list_of_probes
        pass

    def get_alignment_target(self):
        raise NotImplementedError

    def to_dict(self):
        self.lsit_of_probes

    def data(self):
        self.to_dict()

    def iterate_probes(self):
        return iter(map(ProbeInterface, self.list_of_probes))


class CSVDatasetProbe(ProbeInterface):

    def __init__(self, probe):
        self.probe = probe

    def to_dict(self):
        return self.probe

    def data(self):
        return self.to_dict()

    def respond(self, response_data):
        # no-op
        pass

    def pretty_print_str(self):
        return json.dumps(self.data(), indent=2)
