from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from align_system.interfaces.abstracts import Interface, ActionBasedScenarioInterface
from align_system.interfaces.ai2thor_env import AI2ThorEnv
from align_system.data_models.ai2thor import Action as PlannerAction


@dataclass
class _AI2ThorMetaInfo:
    """Stub that satisfies itm_phase1 driver's meta_info.scene_id access."""
    scene_id: str = "ai2thor"


@dataclass
class AI2ThorState:
    """Minimal state object that looks like an ITM State to the driver."""
    unstructured: str
    scenario_complete: bool = False
    env_step: int = field(default=-1)
    meta_info: _AI2ThorMetaInfo = field(default_factory=_AI2ThorMetaInfo)
    # Stubs to satisfy itm_phase1 driver attribute access (unused by AI2Thor)
    characters: List[Any] = field(default_factory=list)

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
    plan: List[PlannerAction] = field(default_factory=list)

    def to_dict(self):
        return {
            "action_id": self.action_id,
            "unstructured": self.unstructured,
            "args": self.args,
        }


class AI2ThorScenario(ActionBasedScenarioInterface):
    def __init__(self, env: AI2ThorEnv, task_spec: Dict[str, Any], scenario_id: str):
        self.env = env
        self.task_spec = task_spec
        self.task = task_spec["description"]
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
            obs = self.env.reset(self.task_spec)
            self._state = AI2ThorState(
                unstructured=f"{self.task}\n\n{obs.text}",
                scenario_complete=False,
                env_step=self.env._step_count,
            )
        return self._state

    def get_available_actions(self) -> List[AI2ThorAction]:
        return [
            AI2ThorAction(action_id=t.name, unstructured=t.description)
            for t in self.env.tools()
        ]

    def take_action(self, action: AI2ThorAction) -> AI2ThorState:
        steps = action.plan if action.plan else [PlannerAction(tool_name=action.action_id, args=action.args or {})]
        result = None
        for planner_action in steps:
            result = self.env.step(planner_action)
            self._state = AI2ThorState(
                unstructured=f"{self.task}\n\n{result.obs.text}",
                scenario_complete=result.done,
                env_step=self.env._step_count,
            )
            if result.done:
                break
        return self._state

    def intend_action(self, action: AI2ThorAction) -> AI2ThorState:
        return self.take_action(action)


class AI2ThorInterface(Interface):
    def __init__(
        self,
        task_definitions: Dict[str, Dict[str, Any]],
        tasks: List[str] = None,
        scene: str = "FloorPlan1",
        starting_points: Dict[str, Any] = None,
        starting_point: str = "default",
        save_frames: bool = False,
        frame_dir: str = "frames",
        **kwargs,
    ):
        self.task_definitions = task_definitions
        self.scene = scene
        self.starting_points = starting_points or {}
        self.starting_point = starting_point
        self.save_frames = save_frames
        self.frame_dir = frame_dir

        tasks = tasks if tasks is not None else ["default"]
        self._queue = [tasks] if isinstance(tasks, str) else list(tasks)

        self._env: Optional[AI2ThorEnv] = None

    def _get_env(self) -> AI2ThorEnv:
        if self._env is None:
            self._env = AI2ThorEnv(
                scene=self.scene,
                save_frames=self.save_frames,
                frame_dir=self.frame_dir,
            )
        return self._env

    def start_scenario(self) -> Optional[AI2ThorScenario]:
        if not self._queue:
            return None
        task_name = self._queue.pop(0)
        if task_name not in self.task_definitions:
            raise ValueError(
                f"Unknown task '{task_name}'; available tasks: "
                f"{list(self.task_definitions)}")

        task_spec = dict(self.task_definitions[task_name])
        # Tasks with a procedural scene setup control their own agent
        # placement; otherwise apply the configured starting point
        if 'setup' not in task_spec:
            task_spec['start_pose'] = self.starting_points.get(self.starting_point)

        env = self._get_env()
        scenario_id = f"{self.scene}-{task_name}"
        return AI2ThorScenario(env=env, task_spec=task_spec, scenario_id=scenario_id)

    def get_session_alignment(self, alignment_target):
        return None
