from abc import ABC, abstractmethod

from swagger_client.models import State, Action, AlignmentTarget


class ActionBasedADM(ABC):
    @abstractmethod
    def choose_action(self,
                      scenario_state: State,
                      available_actions: list[Action],
                      alignment_target: type[AlignmentTarget | None],
                      **kwargs) -> Action:
        pass
