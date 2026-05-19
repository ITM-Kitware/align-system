from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict

JSON = Dict[str, Any]

@dataclass
class PlanCandidate:
    actions: List[Action]
    rationale: str = ""


@dataclass(frozen=True)
class Observation:
    text: str
    raw: Optional[JSON] = None  # full simulator metadata/event snapshot


@dataclass(frozen=True)
class Action:
    tool_name: str
    args: JSON


@dataclass
class StepResult:
    obs: Observation
    reward: float
    done: bool
    info: JSON


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    json_schema: JSON
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
