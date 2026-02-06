from abc import ABC, abstractmethod

from typing import Union, Any, Iterable
from swagger_client.models import State, Action, AlignmentTarget


class ActionBasedADM(ABC):
    @abstractmethod
    def choose_action(self,
                      scenario_state: State,
                      available_actions: list[Action],
                      alignment_target: Union[type[AlignmentTarget], None],
                      **kwargs) -> Union[Action, tuple[Action, dict]]:
        pass


class StructuredInferenceEngine(ABC):
    @abstractmethod
    def dialog_to_prompt(self, dialog: list[dict]) -> str:
        pass

    @abstractmethod
    def run_inference(self, prompts: Union[str, list[str]],
                      schema: str) -> Union[dict, list[dict]]:
        pass


class ADMComponent(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def run_returns(self) -> Union[str, Iterable[str]]:
        '''
        This method should return string identifiers for each of the
        returns expect from the `run` method
        '''
        pass
