from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from align_system.interfaces.abstracts import Interface, ActionBasedScenarioInterface
from align_system.algorithms.mcts_adm.ai2thor_env import AI2ThorEnv
from align_system.algorithms.mcts_adm.types import Action as MCTSAction


TASKS = {
    0: "Pick up an Apple.",
    1: "Pick up a Tomato.",
    2: "Pick up the Red Fruit.",
}


@dataclass
class AI2ThorState:
    """Minimal state object that looks like an ITM State to the driver."""
    unstructured: str
    scenario_complete: bool = False

    def to_dict(self):
        return {"unstructured": self.unstructured, "scenario_complete": self.scenario_complete}


@dataclass
class AI2ThorAction:
    """Minimal action object that looks like an ITM Action to the driver."""
    action_id: str
    unstructured: str
    args: Dict[str, Any] = field(default_factory=dict)
    justification: Optional[str] = None
    kdma_association: Optional[Dict[str, Any]] = None
    plan: List[MCTSAction] = field(default_factory=list)

    def to_dict(self):
        return {
            "action_id": self.action_id,
            "unstructured": self.unstructured,
            "args": self.args,
        }


class AI2ThorScenario(ActionBasedScenarioInterface):
    def __init__(self, env: AI2ThorEnv, task: str, scenario_id: str):
        self.env = env
        self.task = task
        self._scenario_id = scenario_id
        self._state: Optional[AI2ThorState] = None

    def id(self) -> str:
        return self._scenario_id

    def get_alignment_target(self):
        return None

    def to_dict(self):
        return {"task": self.task, "scenario_id": self._scenario_id}

    def data(self):
        return self

    def get_state(self) -> AI2ThorState:
        if self._state is None:
            obs = self.env.reset(self.task)
            self._state = AI2ThorState(unstructured=obs.text, scenario_complete=False)
        return self._state

    def get_available_actions(self) -> List[AI2ThorAction]:
        return [
            AI2ThorAction(action_id=t.name, unstructured=t.description)
            for t in self.env.tools()
        ]

    def take_action(self, action: AI2ThorAction) -> AI2ThorState:
        mcts_action = MCTSAction(tool_name=action.action_id, args=action.args or {})
        result = self.env.step(mcts_action)
        self._state = AI2ThorState(
            unstructured=result.obs.text,
            scenario_complete=result.done,
        )
        return self._state

    def intend_action(self, action: AI2ThorAction) -> AI2ThorState:
        return self.take_action(action)


class AI2ThorInterface(Interface):
    def __init__(
        self,
        scene: str = "FloorPlan1",
        prompts: List[int] = None,
        save_frames: bool = False,
        frame_dir: str = "frames",
        starting_point: str = "default",
        **kwargs,
    ):
        self.scene = scene
        self.save_frames = save_frames
        self.frame_dir = frame_dir
        self.starting_point = starting_point

        prompts = prompts if prompts is not None else [0]
        self._queue = list(prompts)

        self._env: Optional[AI2ThorEnv] = None

    def _get_env(self, prompt: int) -> AI2ThorEnv:
        if self._env is None:
            self._env = AI2ThorEnv(
                scene=self.scene,
                prompt=prompt,
                save_frames=self.save_frames,
                frame_dir=self.frame_dir,
                starting_point=self.starting_point,
            )
        else:
            self._env.prompt = prompt
        return self._env

    def start_scenario(self) -> Optional[AI2ThorScenario]:
        if not self._queue:
            return None
        prompt = self._queue.pop(0)
        task = TASKS.get(prompt, TASKS[0])
        env = self._get_env(prompt)
        scenario_id = f"{self.scene}-prompt{prompt}"
        return AI2ThorScenario(env=env, task=task, scenario_id=scenario_id)

    def get_session_alignment(self, alignment_target):
        return None
