from abc import ABC, abstractmethod


class Interface(ABC):
    @abstractmethod
    def start_scenario(self):
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


class ActionBasedScenarioInterface(ABC):
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
    def get_available_actions(self):
        pass

    @abstractmethod
    def intend_action(self):
        pass

    @abstractmethod
    def take_action(self):
        pass

    @abstractmethod
    def get_state(self):
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
