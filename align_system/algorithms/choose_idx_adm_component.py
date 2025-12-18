from align_system.algorithms.abstracts import ADMComponent

class ChooseIdxADMComponent(ADMComponent):
    def __init__(self, choice_idx=0):
        if choice_idx < 0:
            raise RuntimeError(f"Choice index must be nonnegative, received: {choice_idx}")
        self.choice_idx = choice_idx

    def run_returns(self):
        return 'chosen_choice', 'justification'

    def run(self, choices):

        return choices[self.choice_idx % len(choices)], f"Always choose choice index {self.choice_idx}"
