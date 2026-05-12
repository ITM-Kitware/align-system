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

from .itm_adapter import (
    actions_to_toolspecs,
    alignment_target_to_task,
    find_itm_action,
    state_to_observation,
)
from .llm_ollama import OllamaConfig, OllamaITMCritic, OllamaITMProposer
from .planner import MCTSPlanner


class MCTSActionBasedADM(ActionBasedADM):
    def __init__(
        self,
        model: str = "gpt-oss:20b",
        temperature: float = 0.7,
        num_ctx: int = 8192,
        expansions: int = 5,
        proposals_per_expand: int = 4,
        rollout_horizon: int = 1,
        uct_c: float = 1.4,
        seed: int = 0,
    ):
        prop_cfg = OllamaConfig(model=model, temperature=temperature, num_ctx=num_ctx)
        crit_cfg = OllamaConfig(model=model, temperature=0.0, num_ctx=num_ctx)

        self.planner = MCTSPlanner(
            proposer=OllamaITMProposer(prop_cfg),
            critic=OllamaITMCritic(crit_cfg),
            expansions=expansions,
            proposals_per_expand=proposals_per_expand,
            rollout_horizon=rollout_horizon,
            uct_c=uct_c,
            seed=seed,
        )

    def choose_action(
        self,
        scenario_state,
        available_actions: List,
        alignment_target=None,
        **kwargs,
    ) -> Union[object, tuple]:
        obs = state_to_observation(scenario_state)
        tools = actions_to_toolspecs(available_actions)
        task = alignment_target_to_task(alignment_target, scenario_state)

        log.info(f"[MCTS ADM] task: {task}")
        log.debug(f"[MCTS ADM] observation:\n{obs.text}")

        plan = self.planner.plan(
            task=task,
            tasks_to_consider=task,
            obs=obs,
            tools=tools,
            action_history=[],
        )

        if not plan:
            log.warning("[MCTS ADM] planner returned no actions; falling back to first available")
            chosen = available_actions[0]
        else:
            chosen_id = plan[0].tool_name
            chosen = find_itm_action(chosen_id, available_actions)
            if chosen is None:
                log.warning(f"[MCTS ADM] action_id '{chosen_id}' not found; falling back to first")
                chosen = available_actions[0]
            else:
                log.info(f"[MCTS ADM] chose: {chosen_id}")

        return chosen, {}
