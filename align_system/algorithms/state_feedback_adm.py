from typing import List, Dict, Any

from align_system.algorithms.abstracts import ActionBasedADM, ADMComponent

class InMemoryState():
    def __init__(self):
        self.state = []
    def save(self, state: str) -> None:
        self.state.append(state)
    def retreive(self) -> List[Any]:
        return self.state

class FullStateInMemorySaverADM(ADMComponent):
    def __init__(self, state: InMemoryState):
        self.state = state
    def run(self, state):
        self.state.save(state)
    def run_returns(self):
        return ""


class PassthroughStateRetriever(ADMComponent):
    def __init__(self, state: InMemoryState):
            self.state = state
    def run(self) -> Dict[str, List[str]]:
        return {"previous_state": self.state.retreive()}
    def run_returns(self):
        return ["previous_state"]

# hydra to compose saver and retrieve with the same memorystate
# hydra to create the pipeline