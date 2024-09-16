from abc import ABC, abstractmethod

from typing import Union, Tuple, Dict
from swagger_client.models import State, Action, AlignmentTarget


class ActionBasedADM(ABC):
    @abstractmethod
    def choose_action(self,
                      scenario_state: State,
                      available_actions: list[Action],
                      alignment_target: Union[type[AlignmentTarget], None],
                      **kwargs) -> Union[Action, Tuple[Action, Dict]]:
        pass


# ADM sub-classes implement all the algorithm-specific logic
class AlignedDecisionMaker:
    @abstractmethod
    def __call__(self, sample, target_kdma_values, **kwargs):

        '''
        target_kdma_values: {
            kdma_name: kdma_value,
            ...
        }

        sample = {
            scenario,
            state,
            probe,
            choices: [
                choice_text,
                ...
            ]
        }

        returns {
            choice: idx, [required]
            predicted_kdmas: { [optional]
                0: {
                   kdma_name: kdma_value,
                },
                1: { ... }
            }
        }
        '''
        pass
