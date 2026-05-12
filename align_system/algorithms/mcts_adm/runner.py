from __future__ import annotations
from typing import List, Protocol

from .planner import MCTSPlanner
from .types import Action, Observation, StepResult, ToolSpec


class EnvAdapter(Protocol):
    def reset(self, task: str) -> Observation: ...
    def tools(self) -> List[ToolSpec]: ...
    def step(self, action: Action) -> StepResult: ...


class Runner:
    def __init__(
        self,
        env: EnvAdapter,
        planner: MCTSPlanner,
        max_steps: int = 50,
        verbose: bool = True,
        max_exec_per_plan: int = 1, 
        break_on_failure: bool = True,
    ):
        self.env = env
        self.planner = planner
        self.max_steps = max_steps
        self.verbose = verbose
        self.max_exec_per_plan = max_exec_per_plan
        self.break_on_failure = break_on_failure

    def run(self, task: str, tasks_to_consider: str) -> List[Action]:
        obs = self.env.reset(task)
        tools = self.env.tools()
        history: List[Action] = []

        # Count executed actions (not planner calls) against max_steps
        t = 0
        while t < self.max_steps:
            plan_actions = self.planner.plan(task,tasks_to_consider, obs, tools, history)
            if not plan_actions:
                if self.verbose:
                    print(f"[t={t}] planner returned no action; stopping.")
                break

            # Execute a short prefix of the plan
            exec_actions = plan_actions[: self.max_exec_per_plan]

            if self.verbose:
                pretty = " -> ".join([a.tool_name for a in exec_actions])
                print(f"[t={t}] PLAN({len(plan_actions)}): {pretty}")

            for a in exec_actions:
                if t >= self.max_steps:
                    break

                if self.verbose:
                    print(f"[t={t}] ACTION: {a.tool_name} {a.args}")

                sr = self.env.step(a)
                history.append(a)
                obs = sr.obs
                t += 1

                if self.verbose:
                    print(f"  success={sr.info.get('success')} reward={sr.reward} done={sr.done}")
                    if not sr.info.get("success", True):
                        print(f"  error={sr.info.get('error')}")

                if sr.done:
                    if self.verbose:
                        print("Task done.")
                    return history

                # If an action failed, optionally replan immediately
                if self.break_on_failure and not sr.info.get("success", True):
                    if self.verbose:
                        print("  action failed; breaking out to replan.")
                        
                    break

        return history
