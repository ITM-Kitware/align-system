from __future__ import annotations

import logging as _logging
from typing import List, Union

try:
    from align_system.algorithms.abstracts import ActionBasedADM
    from align_system.utils import logging
    log = logging.getLogger(__name__)
except ModuleNotFoundError:
    ActionBasedADM = object
    log = _logging.getLogger(__name__)

from .llm_ollama import OllamaAI2ThorCritic, OllamaAI2ThorProposer, OllamaConfig
from .planner import MCTSPlanner
from .types import Action, Observation, ToolSpec


class MCTSA2ThorADM(ActionBasedADM):
    """
    ActionBasedADM for AI2Thor that maintains action history across
    choose_action calls within a single scenario run.
    """

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        temperature: float = 0.7,
        num_ctx: int = 8192,
        expansions: int = 5,
        proposals_per_expand: int = 4,
        rollout_horizon: int = 3,
        uct_c: float = 1.4,
        seed: int = 0,
    ):
        prop_cfg = OllamaConfig(
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            max_actions_per_plan=rollout_horizon,
        )
        crit_cfg = OllamaConfig(model=model, temperature=0.0, num_ctx=num_ctx)

        self.planner = MCTSPlanner(
            proposer=OllamaAI2ThorProposer(prop_cfg),
            critic=OllamaAI2ThorCritic(crit_cfg),
            expansions=expansions,
            proposals_per_expand=proposals_per_expand,
            rollout_horizon=rollout_horizon,
            uct_c=uct_c,
            seed=seed,
        )
        self._history: List[Action] = []

    def reset_history(self) -> None:
        self._history = []

    def choose_action(
        self,
        scenario_state,
        available_actions: List,
        alignment_target=None,
        **kwargs,
    ) -> Union[object, tuple]:
        obs = Observation(text=scenario_state.unstructured)

        tools = [
            ToolSpec(
                name=a.action_id,
                description=a.unstructured,
                json_schema={"type": "object", "properties": {}, "required": []},
            )
            for a in available_actions
        ]

        task = scenario_state.unstructured

        plan = self.planner.plan(
            task=task,
            tasks_to_consider=task,
            obs=obs,
            tools=tools,
            action_history=self._history,
        )

        tool_map = {a.action_id: a for a in available_actions}

        if not plan:
            log.warning("[MCTS AI2Thor] planner returned no actions; falling back to first")
            chosen = available_actions[0]
        else:
            chosen_tool = plan[0].tool_name
            chosen = tool_map.get(chosen_tool)
            if chosen is None:
                log.warning(f"[MCTS AI2Thor] '{chosen_tool}' not in available actions; falling back")
                chosen = available_actions[0]
            else:
                self._history.append(plan[0])
                log.info(f"[MCTS AI2Thor] chose: {chosen_tool}")

        return chosen, {}
