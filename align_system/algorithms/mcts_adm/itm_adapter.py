from __future__ import annotations

from typing import Any, List, Optional

from .types import Observation, ToolSpec


def state_to_observation(scenario_state: Any) -> Observation:
    """Convert an ITM scenario State into a text Observation for the MCTS planner."""
    parts: List[str] = []

    unstructured = getattr(scenario_state, "unstructured", None)
    if unstructured:
        parts.append(f"SCENARIO: {unstructured}")

    characters = getattr(scenario_state, "characters", None) or []
    if characters:
        parts.append("CASUALTIES:")
        for c in characters:
            name = getattr(c, "name", None) or getattr(c, "id", "unknown")
            line = f"  - {name}"
            unstructured_c = getattr(c, "unstructured", None)
            if unstructured_c:
                line += f": {unstructured_c}"
            injuries = getattr(c, "injuries", None) or []
            if injuries:
                injury_texts = []
                for inj in injuries:
                    inj_name = getattr(inj, "name", None) or getattr(inj, "type", "injury")
                    location = getattr(inj, "location", None)
                    severity = getattr(inj, "severity", None)
                    desc = inj_name
                    if location:
                        desc += f" at {location}"
                    if severity:
                        desc += f" (severity: {severity})"
                    injury_texts.append(desc)
                line += f" — injuries: {', '.join(injury_texts)}"
            parts.append(line)

    supplies = getattr(scenario_state, "supplies", None) or []
    if supplies:
        supply_strs = []
        for s in supplies:
            stype = getattr(s, "type", None) or str(s)
            qty = getattr(s, "quantity", None)
            supply_strs.append(f"{stype}{'×'+str(qty) if qty is not None else ''}")
        parts.append(f"SUPPLIES: {', '.join(supply_strs)}")

    environment = getattr(scenario_state, "environment", None)
    if environment:
        unstructured_env = getattr(environment, "unstructured", None)
        if unstructured_env:
            parts.append(f"ENVIRONMENT: {unstructured_env}")

    return Observation(text="\n".join(parts))


def actions_to_toolspecs(available_actions: List[Any]) -> List[ToolSpec]:
    """Convert ITM available actions into MCTS ToolSpecs."""
    specs = []
    for action in available_actions:
        action_id = getattr(action, "action_id", None) or str(action)
        description = getattr(action, "unstructured", None) or action_id
        specs.append(ToolSpec(
            name=action_id,
            description=description,
            json_schema={"type": "object", "properties": {}, "required": []},
        ))
    return specs


def find_itm_action(action_id: str, available_actions: List[Any]) -> Optional[Any]:
    """Return the ITM Action object whose action_id matches, or None."""
    for action in available_actions:
        if getattr(action, "action_id", None) == action_id:
            return action
    return None


def alignment_target_to_task(alignment_target: Any, scenario_state: Any) -> str:
    """Build a human-readable task string from the ITM alignment target."""
    if alignment_target is None:
        unstructured = getattr(scenario_state, "unstructured", "")
        return f"Choose the best action for this medical triage scenario: {unstructured}"

    kdma_values = getattr(alignment_target, "kdma_values", None) or []
    if not kdma_values:
        return "Choose the best medically appropriate action."

    kdma_parts = []
    for kv in kdma_values:
        kdma = getattr(kv, "kdma", None)
        value = getattr(kv, "value", None)
        if kdma and value is not None:
            level = "high" if value >= 0.5 else "low"
            kdma_parts.append(f"{level} {kdma}")

    unstructured = getattr(scenario_state, "unstructured", "")
    if kdma_parts:
        return (
            f"Choose the best action for this medical triage scenario, "
            f"prioritizing: {', '.join(kdma_parts)}. Scenario: {unstructured}"
        )
    return f"Choose the best action for this medical triage scenario: {unstructured}"
