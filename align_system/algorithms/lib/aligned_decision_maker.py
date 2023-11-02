from abc import abstractmethod

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