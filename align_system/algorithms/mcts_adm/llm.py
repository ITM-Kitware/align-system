from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol, Optional, Tuple
from .types import Observation, Action, ToolSpec


@dataclass
class PlanCandidate:
    actions: List[Action]
    rationale: str = ""


class ProposerLLM(Protocol):
    def propose(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[Action],
        k: int,
        diversity_hint: Optional[str] = None,
    ) -> List[PlanCandidate]:
        ...


class CriticLLM(Protocol):
    def score(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[Action],
        proposed_future: List[Action],
        rationale: str = "",
    ) -> Tuple[float, str]:
        """Return (score, reason) where higher score is better."""
        ...
