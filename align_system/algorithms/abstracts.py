from abc import ABC, abstractmethod

from typing import Union, Tuple, Dict, List
from swagger_client.models import State, Action, AlignmentTarget

from align_system.data_models.dialog import Dialog


class ActionBasedADM(ABC):
    @abstractmethod
    def choose_action(self,
                      scenario_state: State,
                      available_actions: list[Action],
                      alignment_target: Union[type[AlignmentTarget], None],
                      **kwargs) -> Union[Action, Tuple[Action, Dict]]:
        pass


class StructuredInferenceEngine(ABC):
    @abstractmethod
    def dialog_to_prompt(dialog: list[Dict]) -> str:
        pass

    @abstractmethod
    def run_inference(prompts: list[str],
                      schema: str) -> list[Dict]:
        pass


class ADMComponent(ABC):
    @abstractmethod
    def run(self,
            scenario_state: State,
            choice_evaluation: Dict,
            dialogs: List[Dialog],
            alignment_target: Union[type[AlignmentTarget], None]) -> Tuple[Dict, List[Dialog]]:
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
