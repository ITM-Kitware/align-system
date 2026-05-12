from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ollama

from .llm import PlanCandidate, ProposerLLM, CriticLLM
from .types import Action, Observation, ToolSpec

JSON = Dict[str, Any]


def _extract_json_object(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError("No JSON object found")
    return s[i : j + 1]


def _loads_json(s: str) -> JSON:
    return json.loads(_extract_json_object(s))


def _repair_json(model: str, bad_text: str, schema_hint: str, num_ctx: int) -> JSON:
    prompt = (
        "You output invalid JSON. Fix it.\n"
        "Return ONLY valid JSON, no prose.\n"
        f"Schema hint:\n{schema_hint}\n\n"
        f"Bad output:\n{bad_text}\n"
    )
    resp = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.0, "num_ctx": num_ctx})
    return _loads_json(resp["response"])


def _tool_compact(tools: List[ToolSpec]) -> List[JSON]:
    return [{"name": t.name, "description": t.description} for t in tools]


@dataclass
class OllamaConfig:
    model: str = "gpt-oss:20b"
    temperature: float = 0.7
    num_ctx: int = 8192
    json_repair_attempts: int = 1
    max_actions_per_plan: int = 1


class OllamaITMProposer(ProposerLLM):
    def __init__(self, cfg: OllamaConfig = OllamaConfig()):
        self.cfg = cfg

    def propose(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[Action],
        k: int,
        diversity_hint: Optional[str] = None,
    ) -> List[PlanCandidate]:
        schema_hint = '{"candidates":[{"actions":[{"tool_name":"<action_id>","args":{}}],"rationale":"..."}]}'

        payload = {
            "task": task,
            "scenario": obs.text,
            "available_actions": _tool_compact(tools),
            "recent_choices": [a.tool_name for a in action_history[-5:]],
            "k": k,
        }

        prompt = (
            "You are a medical triage decision-making assistant.\n"
            "Return ONLY valid JSON matching the output schema. No extra text.\n\n"
            "Given the scenario and task, propose k candidate action choices.\n"
            "Each candidate must use exactly ONE action from available_actions.\n"
            "The tool_name must be copied exactly from an available_actions name.\n"
            "Vary your reasoning across candidates; consider different ethical trade-offs.\n"
            f"{diversity_hint or ''}\n\n"
            f"INPUT:\n{json.dumps(payload)}\n\n"
            f"OUTPUT schema:\n{schema_hint}\n"
        )

        resp = ollama.generate(
            model=self.cfg.model,
            prompt=prompt,
            options={"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx},
        )
        text = resp["response"]

        try:
            data = _loads_json(text)
        except Exception:
            data = None
            for _ in range(self.cfg.json_repair_attempts):
                try:
                    data = _repair_json(self.cfg.model, text, schema_hint, self.cfg.num_ctx)
                    break
                except Exception:
                    data = None
            if data is None:
                return []

        tool_names = {t.name for t in tools}
        cands: List[PlanCandidate] = []
        for c in data.get("candidates", [])[:k]:
            actions = []
            for a in c.get("actions", []):
                name = a.get("tool_name", "")
                if name in tool_names:
                    actions.append(Action(tool_name=name, args=a.get("args", {})))
            if actions:
                cands.append(PlanCandidate(actions=actions, rationale=c.get("rationale", "")))

        return cands


class OllamaITMCritic(CriticLLM):
    def __init__(self, cfg: OllamaConfig = OllamaConfig(model="gpt-oss:20b", temperature=0.0)):
        self.cfg = cfg

    def score(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[Action],
        proposed_future: List[Action],
        rationale: str = "",
    ) -> Tuple[float, str]:
        schema_hint = '{"score": 0.0, "reason": "..."}'

        # Resolve proposed action names to descriptions for the critic
        tool_map = {t.name: t.description for t in tools}
        proposed_descriptions = [
            tool_map.get(a.tool_name, a.tool_name) for a in proposed_future
        ]

        payload = {
            "task": task,
            "scenario": obs.text,
            "proposed_actions": proposed_descriptions,
            "recent_choices": [a.tool_name for a in action_history[-5:]],
        }

        prompt = (
            "You are a strict evaluator of medical triage decisions.\n"
            "Return ONLY valid JSON. No extra text.\n\n"
            "Score how well the proposed action addresses the task considering:\n"
            "- Medical appropriateness and urgency\n"
            "- Alignment with the stated ethical/value target\n"
            "- Feasibility given the scenario\n"
            "Use a score from 0.0 (completely wrong) to 10.0 (ideal choice).\n\n"
            f"INPUT:\n{json.dumps(payload)}\n\n"
            f"OUTPUT schema:\n{schema_hint}\n"
        )

        resp = ollama.generate(
            model=self.cfg.model,
            prompt=prompt,
            options={"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx},
        )
        text = resp["response"]

        try:
            data = _loads_json(text)
        except Exception:
            data = None
            for _ in range(self.cfg.json_repair_attempts):
                try:
                    data = _repair_json(self.cfg.model, text, schema_hint, self.cfg.num_ctx)
                    break
                except Exception:
                    data = None
            if data is None:
                return 0.0, ""

        try:
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")
            return score, reason
        except Exception:
            return 0.0, ""


# ---------------------------------------------------------------------------
# AI2Thor-specific proposer / critic
# ---------------------------------------------------------------------------

class OllamaAI2ThorProposer(ProposerLLM):
    """Proposer with prompts tailored for embodied navigation in AI2Thor."""

    def __init__(self, cfg: OllamaConfig = OllamaConfig()):
        self.cfg = cfg

    def propose(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[Action],
        k: int,
        diversity_hint: Optional[str] = None,
    ) -> List[PlanCandidate]:
        schema_hint = (
            '{"candidates":[{"actions":[{"tool_name":"MoveAhead","args":{"moveMagnitude":0.25}}],'
            '"rationale":"..."}]}'
        )

        payload = {
            "task": task,
            "observation": obs.text,
            "recent_actions": [{"tool_name": a.tool_name, "args": a.args} for a in action_history[-10:]],
            "tools": [{"name": t.name, "description": t.description, "schema": t.json_schema} for t in tools],
            "k": k,
            "max_actions_per_plan": self.cfg.max_actions_per_plan,
            "diversity_hint": diversity_hint or "",
        }

        prompt = (
            "You are an embodied planning model.\n"
            "Return ONLY valid JSON. No extra text.\n"
            "Generate k semi-diverse candidate plans.\n"
            "- Each plan is 1 to max_actions_per_plan actions.\n"
            "- Use ONLY the tool names provided.\n"
            "- Args MUST satisfy each tool schema.\n"
            "- IMPORTANT objectId rule: For tools requiring objectId (TeleportNearObject, PickupObject, "
            "OpenObject, CloseObject, ToggleObjectOn/Off), you MUST copy the exact full objectId string "
            "from the observation's visible lines (the value after 'id='). "
            "Never use object type names like 'Apple' as objectId. Full objectIds contain '|' characters.\n"
            "- Avoid repeating the same last action unless clearly helpful.\n"
            f"{diversity_hint or ''}\n\n"
            f"INPUT:\n{json.dumps(payload)}\n\n"
            f"OUTPUT schema:\n{schema_hint}\n"
        )

        resp = ollama.generate(
            model=self.cfg.model,
            prompt=prompt,
            options={"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx},
        )
        text = resp["response"]

        try:
            data = _loads_json(text)
        except Exception:
            data = None
            for _ in range(self.cfg.json_repair_attempts):
                try:
                    data = _repair_json(self.cfg.model, text, schema_hint, self.cfg.num_ctx)
                    break
                except Exception:
                    data = None
            if data is None:
                return [PlanCandidate(actions=[Action("RotateRight", {})], rationale="Fallback scan.")]

        tool_names = {t.name for t in tools}
        cands: List[PlanCandidate] = []
        for c in data.get("candidates", [])[:k]:
            actions = []
            for a in c.get("actions", []):
                name = a.get("tool_name", "")
                if name in tool_names:
                    actions.append(Action(tool_name=name, args=a.get("args", {}) or {}))
            if actions:
                cands.append(PlanCandidate(actions=actions, rationale=c.get("rationale", "")))

        if not cands:
            cands = [PlanCandidate(actions=[Action("RotateRight", {})], rationale="Fallback scan.")]
        return cands


class OllamaAI2ThorCritic(CriticLLM):
    """Critic with prompts tailored for embodied navigation in AI2Thor."""

    def __init__(self, cfg: OllamaConfig = OllamaConfig(model="gpt-oss:20b", temperature=0.0)):
        self.cfg = cfg

    def score(
        self,
        task: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[Action],
        proposed_future: List[Action],
        rationale: str = "",
    ) -> Tuple[float, str]:
        schema_hint = '{"score": 0.0, "reason": "..."}'

        payload = {
            "task": task,
            "observation": obs.text,
            "recent_actions": [{"tool_name": a.tool_name, "args": a.args} for a in action_history[-10:]],
            "proposed_future": [{"tool_name": a.tool_name, "args": a.args} for a in proposed_future],
        }

        prompt = (
            "You are a strict critic for embodied plans.\n"
            "Return ONLY valid JSON. No extra text.\n"
            "Score how good the proposed_future is for achieving the task from the observation.\n"
            "Use a score from 0.0 to 10.0.\n\n"
            "Penalize any plan that uses an objectId that is not a full AI2-THOR id containing '|'.\n"
            "Feasibility matters: PickupObject requires a valid visible objectId.\n\n"
            f"INPUT:\n{json.dumps(payload)}\n\n"
            f"OUTPUT schema:\n{schema_hint}\n"
        )

        resp = ollama.generate(
            model=self.cfg.model,
            prompt=prompt,
            options={"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx},
        )
        text = resp["response"]

        try:
            data = _loads_json(text)
        except Exception:
            data = None
            for _ in range(self.cfg.json_repair_attempts):
                try:
                    data = _repair_json(self.cfg.model, text, schema_hint, self.cfg.num_ctx)
                    break
                except Exception:
                    data = None
            if data is None:
                return 0.0, ""

        try:
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")
            print(f"[CRITIC] score={score:.3f}  reason: {reason}")
            return score, reason
        except Exception:
            return 0.0, ""
