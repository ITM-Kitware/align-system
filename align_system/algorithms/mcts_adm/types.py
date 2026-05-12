from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

JSON = Dict[str, Any]


@dataclass(frozen=True)
class Observation:
    text: str
    raw: Optional[JSON] = None


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
