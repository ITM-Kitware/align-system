import json

from align_system.interfaces.abstracts import (
    Interface,
    ScenarioInterface,
    ProbeInterface)


class LocalFilesInterface(Interface):
    def __init__(self,
                 probes_filepaths,
                 scenario_filepath,
                 alignment_target_filepath=None):
        self._probes_filepaths = probes_filepaths
        self._scenario_filepath = scenario_filepath
        self._alignment_target_filepath = alignment_target_filepath

    def start_scenario(self):
        with open(self._scenario_filepath) as f:
            scenario_data = json.load(f)

        return LocalFilesScenario(
            scenario_data,
            probes_filepaths=self._probes_filepaths,
            alignment_target_filepath=self._alignment_target_filepath)


class LocalFilesScenario(ScenarioInterface):
    def __init__(self,
                 scenario,
                 probes_filepaths=[],
                 alignment_target_filepath=None):
        self._scenario = scenario

        if alignment_target_filepath is None:
            self._alignment_target = None
        else:
            with open(alignment_target_filepath) as f:
                self._alignment_target = json.load(f)

        self._probes = []
        if len(probes_filepaths) > 0:
            for probe_filepath in probes_filepaths:
                with open(probe_filepath) as f:
                    self._probes.append(json.load(f))
        else:
            self._probes.extend(scenario.get('probes', []))

    def get_alignment_target(self):
        if self._alignment_target is None:
            raise RuntimeError(
                "Requested alignment target, but the local file path for "
                "it wasn't provided with `--alignment-target-filepath`")
        else:
            return self._alignment_target

    def to_dict(self):
        return self._scenario

    def data(self):
        return self._scenario

    def iterate_probes(self):
        for probe in self._probes:
            yield LocalFilesProbe(probe)


class LocalFilesProbe(ProbeInterface):
    def __init__(self, probe_data):
        self._probe_data = probe_data

    def to_dict(self):
        return self._probe_data

    def data(self):
        return self._probe_data

    def respond(self, response_data):
        pass

    def pretty_print_str(self):
        probe = self._probe_data

        options_string = "\n".join([f"[{o['id']}] {o['value']}"
                                    for o in probe.get('options', [])])

        return f"[{probe['id']}] {probe['prompt']}\n{options_string}"
