from abc import ABC, abstractmethod


class Interface(ABC):
    @abstractmethod
    def start_scenario(self):
        pass

    @classmethod
    @abstractmethod
    def cli_parser(cls, parser=None):
        pass

    @classmethod
    @abstractmethod
    def cli_parser_description(cls):
        pass

    @classmethod
    @abstractmethod
    def init_from_parsed_args(cls, parsed_args):
        pass


class ScenarioInterface(ABC):
    @abstractmethod
    def get_alignment_target(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def data(self):
        pass

    @abstractmethod
    def iterate_probes(self):
        pass


class ScenarioInterfaceWithAlignment(ScenarioInterface):
    @abstractmethod
    def get_alignment_results(self):
        pass


class ProbeInterface(ABC):
    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def data(self):
        pass

    @abstractmethod
    def respond(self, response_data):
        pass

    @abstractmethod
    def pretty_print_str(self):
        pass


class ProbeInterfaceWithAlignment(ProbeInterface):
    @abstractmethod
    def get_alignment_results(self):
        pass
