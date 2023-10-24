from abc import abstractmethod

# ADM sub-classes implement all the algorithm-specific logic
class AutomatedDecisionMaker:
    
    @abstractmethod
    def __call__(self, sample, **kwargs):
        '''
        sample = {
                target_kdmas: { ... }
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