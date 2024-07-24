from typing import List, TypedDict, Literal


# Some types
class Probe(TypedDict):
    probe: str
    probe_prompt: str
    response: str
    response_value: float


class Backstory(TypedDict):
    backstory: str
    probes: List[Probe]


class DialogTurn(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

Dialog = List[DialogTurn]
