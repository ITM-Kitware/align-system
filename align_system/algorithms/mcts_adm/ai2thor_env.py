from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
from pathlib import Path

from PIL import Image
import numpy as np
from ai2thor.controller import Controller

from .types import Action, Observation, StepResult, ToolSpec
import math
from typing import Optional, Dict, List
JSON = Dict[str, Any]


# at module top
from typing import Dict, Any, List
import threading

# module-level cache for seen objects (objectId -> object metadata)
_SEEN_OBJECTS: Dict[str, Dict[str, Any]] = {}
_SEEN_LOCK = threading.Lock()


def _first_visible_object_id_of_type(event, object_type: str) -> Optional[str]:
    for o in event.metadata.get("objects", []):
        if o.get("visible") and o.get("objectType") == object_type:
            return o.get("objectId")
    return None

def reset_seen_objects() -> None:
    """Clear the memory of seen objects. Call at env.reset(...) if you want per-episode memory."""
    global _SEEN_OBJECTS
    with _SEEN_LOCK:
        _SEEN_OBJECTS = {}


def get_seen_objects() -> Dict[str, Dict[str, Any]]:
    """Return a shallow copy of the current seen objects mapping (objectId -> metadata)."""
    with _SEEN_LOCK:
        return dict(_SEEN_OBJECTS)


def _compact_obj_line(o: Dict[str, Any]) -> str:
    """Return a compact single-line summary for one object metadata dict."""
    oid = o.get("objectId", "")
    short = oid.split("|")[0] if oid else "Unknown"
    props: List[str] = []
    if o.get("pickupable"):
        props.append("pickupable")
    if o.get("openable"):
        props.append("openable")
    if o.get("toggleable"):
        props.append("toggleable")
    if o.get("isOpen"):
        props.append("isOpen")
    if o.get("isToggled"):
        props.append("isToggled")
    # Include a short position tail if present (helps disambiguate duplicates)
    pos = o.get("position") or o.get("position", {})  # some metadata uses different keys
    pos_tail = ""
    if isinstance(pos, dict) and pos.get("x") is not None:
        pos_tail = f"@({pos['x']:.2f},{pos['y']:.2f},{pos['z']:.2f})"
    props_str = ("/".join(props) + " ") if props else ""
    return f"- {short} id={oid} {props_str}{pos_tail}".strip()


def _summarize(
    event,
    *,
    update_memory: bool = True,
    max_visible: int = 15,
    max_prev: int = 120,
) -> str:
    """
    Compact observation summary with disjoint object-id sets:
      - visible_ids: objectIds visible right now (for interaction)
      - previously_seen_ids: objectIds seen before but not visible right now (for memory/navigation)
      - visible: compact actionable lines (id + affordances + pos)

    Memory is updated ONLY from objects visible in this event (unless update_memory=False).
    """
    md = getattr(event, "metadata", {}) or {}

    agent = (md.get("agent") or {})
    pos = (agent.get("position") or {})
    rot = (agent.get("rotation") or {})
    yaw = rot.get("y", 0.0)

    inv = md.get("inventoryObjects") or []
    held_types = [i.get("objectType", "Unknown") for i in inv]
    held_ids = [i.get("objectId") for i in inv if i.get("objectId")]

    objs = md.get("objects") or []
    vis = [o for o in objs if o.get("visible")]

    # IDs visible now (canonical for "do it now" interaction tools)
    visible_ids = [o.get("objectId") for o in vis if o.get("objectId")]
    visible_set = set(visible_ids)

    # Update memory ONLY from visible objects
    if update_memory:
        with _SEEN_LOCK:
            for o in vis:
                oid = o.get("objectId")
                if oid:
                    _SEEN_OBJECTS[oid] = o.copy()

    # previously seen = ever seen minus currently visible
    with _SEEN_LOCK:
        seen_ids_all = list(_SEEN_OBJECTS.keys())
    prev_ids = [oid for oid in seen_ids_all if oid not in visible_set]

    # optional: group/sort for readability
    def _type_key(oid: str) -> str:
        return oid.split("|")[0] if "|" in oid else oid
    prev_ids = sorted(prev_ids, key=lambda x: (_type_key(x), x))[:max_prev]

    # Visible: compact actionable lines
    def _affordances(o):
        a = []
        if o.get("pickupable"): a.append("pickup")
        if o.get("openable"): a.append("open")
        if o.get("toggleable"): a.append("toggle")
        if o.get("isOpen"): a.append("isOpen")
        if o.get("isToggled"): a.append("isOn")
        return a

    vis_lines = []
    for o in vis[:max_visible]:
        oid = o.get("objectId", "")
        typ = oid.split("|")[0] if oid else (o.get("objectType") or "Unknown")
        aff = _affordances(o)
        p = o.get("position") or {}
        p_str = ""
        if isinstance(p, dict) and p.get("x") is not None:
            p_str = f" @({p['x']:.2f},{p['y']:.2f},{p['z']:.2f})"
        aff_str = f" [{'|'.join(aff)}]" if aff else ""
        vis_lines.append(f"- {typ} id={oid}{aff_str}{p_str}".strip())

    return "\n".join(
        [
            f"state: pos=({pos.get('x',0.0):.2f},{pos.get('y',0.0):.2f},{pos.get('z',0.0):.2f}) yaw={yaw}",
            f"held: types={held_types if held_types else []} ids={held_ids if held_ids else []}",
            f"previously_seen_ids: {prev_ids}",
            "visible:",
            *(vis_lines if vis_lines else ["- (none)"]),
        ]
    )




def _holding_object_type(event, object_type: str) -> bool:
    inv = event.metadata.get("inventoryObjects", [])
    return any(i.get("objectType") == object_type for i in inv)


@dataclass
class AI2ThorEnv:
    scene: str = "FloorPlan1"
    width: int = 600
    height: int = 600
    gridSize: float = 0.25
    visibilityDistance: float = 1.5
    rotateStepDegrees: int = 90
    renderDepthImage: bool = False
    renderInstanceSegmentation: bool = False
    prompt: int = 1
    save_frames: bool = False
    frame_dir: str = "frames"
    starting_point: str = "default"

    def __post_init__(self):
        self.controller: Optional[Controller] = None
        self._task: str = ""
        self._last_event = None
        self._reachable_positions: list[dict] = []
        self._visited_pose_keys: set[str] = set()
        self._step_count: int = 0

    def _save_frame(self, event, action_name: str) -> None:
        """Save the current RGB frame as a PNG labelled by step and action."""
        if not self.save_frames or event is None:
            return
        frame = getattr(event, "frame", None)
        if frame is None:
            print("[AI2ThorEnv] save_frames=True but event has no frame data")
            return
        Path(self.frame_dir).mkdir(parents=True, exist_ok=True)
        fname = f"step{self._step_count:04d}_{action_name}.png"
        fpath = os.path.join(self.frame_dir, fname)
        Image.fromarray(frame.astype(np.uint8)).save(fpath)
        print(f"[AI2ThorEnv] saved frame: {fpath}")

    def reset(self, task: str) -> Observation:
        reset_seen_objects()
        self._task = task
        if self.controller is None:
            self.controller = Controller(
                scene=self.scene,
                width=self.width,
                height=self.height,
                gridSize=self.gridSize,
                visibilityDistance=self.visibilityDistance,
                rotateStepDegrees=self.rotateStepDegrees,
                renderDepthImage=self.renderDepthImage,
                renderInstanceSegmentation=self.renderInstanceSegmentation,
                platform="CloudRendering"
            )
        else:
            self.controller.reset(scene=self.scene)
        
        if self.starting_point=="direct":
            event = self.controller.step(
            action="Teleport",
            forceAction=True,
            position=dict(x=-1.20, y=1.0, z=-0.25),
            rotation=dict(x=0, y=90, z=0),
            horizon=30,
            standing=True,
            )
        elif self.starting_point=="table":
            event = self.controller.step(
            action="Teleport",
            forceAction=True,
            position=dict(x=1.20, y=1.0, z=0.25),
            rotation=dict(x=0, y=270, z=0),
            horizon=30,
            standing=True,
            )
        elif self.starting_point=="tomato":
            event = self.controller.step(
            action="Teleport",
            forceAction=True,
            position=dict(x=-0.50,y=0.90,z=-1.25),
            rotation=dict(x=0, y=357, z=0),
            horizon=30,
            standing=True,

            )


        self._last_event = self.controller.last_event
        self._last_event = self.controller.last_event

        md = self.last_event().metadata
        types = set(o.get("objectType") for o in md.get("objects", []))
        if self.prompt == 0:
            print("Apple in scene?", "Apple" in types)
        else:
            print("Tomato in scene?", "Tomato" in types)

        print(self.prompt)
        # Cache reachable positions ONCE (do not call step in proposer/critic)
        ev = self.controller.step(action="GetReachablePositions")
        self._reachable_positions = ev.metadata.get("actionReturn", []) or []

        # Seed visited set with initial pose
        self._visited_pose_keys = set()
        self._visited_pose_keys.add(self._pose_key(self._last_event))

        return Observation(text=_summarize(self._last_event), raw=self._last_event.metadata)

    def tools(self) -> List[ToolSpec]:
        # Minimal tool set that is enough to solve simple tasks.
        # AI2-THOR supports navigation + interaction actions. :contentReference[oaicite:2]{index=2}
        return [
            ToolSpec(
                name="PickupObject",
                description="Pick up a visible pickupable object by objectId.",
                json_schema={"type": "object", "properties": {"objectId": {"type": "string"}}, "required": ["objectId"]},
            ),
            ToolSpec(
                name="MoveAhead",
                description="Move forward by moveMagnitude meters.",
                json_schema={"type": "object", "properties": {"moveMagnitude": {"type": "number"}}, "required": []},
            ),
            ToolSpec(
                name="RotateLeft",
                description="Rotate left by rotateStepDegrees.",
                json_schema={"type": "object", "properties": {}, "required": []},
            ),
            ToolSpec(
                name="RotateRight",
                description="Rotate right by rotateStepDegrees.",
                json_schema={"type": "object", "properties": {}, "required": []},
            ),
            ToolSpec(
                name="LookUp",
                description="Look up.",
                json_schema={"type": "object", "properties": {}, "required": []},
            ),
            ToolSpec(
                name="LookDown",
                description="Look down.",
                json_schema={"type": "object", "properties": {}, "required": []},
            ),
            ToolSpec(
                name="DropHandObject",
                description="Drop the object currently held by the agent.",
                json_schema={"type": "object", "properties": {}, "required": []},
            ),

            # ToolSpec(
            #     name="GetReachablePositions",
            #     description="Return reachable positions.",
            #     json_schema={"type": "object", "properties": {}, "required": []},
            # ),
            ToolSpec(
                name="TeleportNearObject",
                description="Teleport to the reachable agent position nearest the given object's position. "
                            "Args: { objectId: <full objectId string> }",
                json_schema={
                    "type": "object",
                    "properties": {"objectId": {"type": "string"}},
                    "required": ["objectId"],
                },
            ),
            # ToolSpec(
            #     name="Teleport",
            #     description="Teleport agent to a position (and optionally rotation/horizon).",
            #     json_schema={
            #         "type": "object",
            #         "properties": {
            #             "position": {
            #                 "type": "object",
            #                 "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
            #                 "required": ["x", "y", "z"],
            #             },
            #             "rotation": {
            #                 "type": "object",
            #                 "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
            #             },
            #             "horizon": {"type": "number"},
            #         },
            #         "required": ["position"],
            #     },
            # ),
            ToolSpec(
                name="PutObject",
                description="Put held object into/on a receptacle by objectId.",
                json_schema={
                    "type": "object",
                    "properties": {"objectId": {"type": "string"}},
                    "required": ["objectId"],
                },
            ),
            ToolSpec(
                name="OpenObject",
                description="Open an openable object by objectId.",
                json_schema={
                    "type": "object",
                    "properties": {"objectId": {"type": "string"}, "openness": {"type": "number"}},
                    "required": ["objectId"],
                },
            ),
            ToolSpec(
                name="CloseObject",
                description="Close an openable object by objectId.",
                json_schema={"type": "object", "properties": {"objectId": {"type": "string"}}, "required": ["objectId"]},
            ),
            ToolSpec(
                name="ToggleObjectOn",
                description="Toggle an object on by objectId.",
                json_schema={"type": "object", "properties": {"objectId": {"type": "string"}}, "required": ["objectId"]},
            ),
            ToolSpec(
                name="ToggleObjectOff",
                description="Toggle an object off by objectId.",
                json_schema={"type": "object", "properties": {"objectId": {"type": "string"}}, "required": ["objectId"]},
            ),
        ]

    def step(self, action: Action) -> StepResult:
        assert self.controller is not None

        #####
        if action.tool_name == "TeleportNearObject":
            obj_id = (action.args or {}).get("objectId")

            if not obj_id or not isinstance(obj_id, str):
                obs = Observation(
                    text="TeleportNearObject failed: missing objectId (must be a full AI2-THOR objectId string).\n"
                        + _summarize(self._last_event),
                    raw=self._last_event.metadata,
                )
                return StepResult(obs=obs, reward=-0.5, done=False, info={"success": False, "event": self._last_event.metadata, "error": "missing objectId"})

            valid_ids = {
                o.get("objectId")
                for o in self._last_event.metadata.get("objects", [])
                if o.get("objectId")
            }

            if obj_id not in valid_ids:
                obs = Observation(
                    text=(
                        "TeleportNearObject blocked: objectId must match an objectId from metadata.\n"
                        f"attempted_objectId={obj_id}\n"
                        "hint: choose an id from visible_objects or call ListVisibleObjects.\n"
                        + _summarize(self._last_event)
                    ),
                    raw=self._last_event.metadata,
                )
                return StepResult(obs=obs, reward=-0.5, done=False, info={"success": False, "event": self._last_event.metadata, "error": "invalid objectId"})

            obj_pos = self._find_object_position(obj_id)
            if obj_pos is None:
                obs = Observation(
                    text=f"TeleportNearObject failed: objectId found but no position for {obj_id}\n" + _summarize(self._last_event),
                    raw=self._last_event.metadata,
                )
                return StepResult(obs=obs, reward=-0.2, done=False, info={"success": False, "event": self._last_event.metadata, "error": "no position"})

            cands = self._k_nearest_reachable_to(obj_pos, k=15)
            if not cands:
                obs = Observation(
                    text="TeleportNearObject failed: no cached reachable positions available.\n" + _summarize(self._last_event),
                    raw=self._last_event.metadata,
                )
                return StepResult(obs=obs, reward=-0.2, done=False, info={"success": False, "event": self._last_event.metadata, "error": "no reachable positions"})

            event = None
            last_err = ""

            # A small downward horizon helps for countertop objects.
            # You can tune this; 30 is a decent default.
            horizon = 30.0

            for r in cands:
                try:
                    yaw = self._yaw_to_face(r, obj_pos)

                    ev = self.controller.step(
                        action="Teleport",
                        position={"x": r["x"], "y": r["y"], "z": r["z"]},
                        rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
                        horizon=float(horizon),
                        forceAction=True,
                    )

                    if bool(ev.metadata.get("lastActionSuccess", False)):
                        event = ev
                        break

                    last_err = ev.metadata.get("errorMessage", "") or last_err
                except Exception as e:
                    last_err = str(e)

            if event is None:
                obs = Observation(
                    text=(
                        "TeleportNearObject failed: all nearby reachable teleports failed (likely collisions).\n"
                        f"last_error={last_err}\n"
                        + _summarize(self._last_event)
                    ),
                    raw=self._last_event.metadata,
                )
                return StepResult(obs=obs, reward=-0.2, done=False, info={"success": False, "event": self._last_event.metadata, "error": last_err})

            self._last_event = event
            self.mark_visited(event)

            # Important: do NOT claim success unconditionally; report the real state
            success = bool(event.metadata.get("lastActionSuccess", False))
            err = event.metadata.get("errorMessage", "")

            obs = Observation(text=_summarize(event), raw=event.metadata)
            if self.prompt == 0:
                done = _holding_object_type(event, "Apple")  # you can keep this for now
            elif self.prompt == 1:
                done = _holding_object_type(event, "Tomato")  # you can keep this for now
            elif self.prompt == 2:
                done = _holding_object_type(event, "Apple") or _holding_object_type(event, "Tomato") or  _holding_object_type(event, "Toaster") or  _holding_object_type(event, "Vase")
  # you can keep this for now
            reward = 10.0 if done else (0.1 if success else -0.2)

            self._save_frame(event, action.tool_name)
            self._step_count += 1
            return StepResult(obs=obs, reward=reward, done=done, info={"success": success, "event": event.metadata, "error": err})

        ########## PickUpObject

        if action.tool_name == "PickupObject":
            obj_id = (action.args or {}).get("objectId")

            # 1) basic validation
            if not obj_id or not isinstance(obj_id, str):
                obs = Observation(
                    text="PickupObject failed: missing objectId (must be a full AI2-THOR objectId string).\n"
                        + _summarize(self._last_event),
                    raw=self._last_event.metadata,
                )
                return StepResult(obs=obs, reward=-0.5, done=False, info={"success": False, "event": self._last_event.metadata, "error": "missing objectId"})

            # optional but recommended: require full objectId formatting
            if "|" not in obj_id:
                obs = Observation(
                    text=(
                        "PickupObject blocked: objectId must be a full AI2-THOR objectId (contains '|').\n"
                        f"attempted_objectId={obj_id}\n"
                        + _summarize(self._last_event)
                    ),
                    raw=self._last_event.metadata,
                )
                return StepResult(obs=obs, reward=-0.5, done=False, info={"success": False, "event": self._last_event.metadata, "error": "invalid objectId"})

            # 2) ensure object exists in metadata
            objs = self._last_event.metadata.get("objects", []) or []
            obj_meta = None
            for o in objs:
                if o.get("objectId") == obj_id:
                    obj_meta = o
                    break

            if obj_meta is None:
                obs = Observation(
                    text=(
                        "PickupObject failed: objectId not found in current metadata.\n"
                        f"attempted_objectId={obj_id}\n"
                        + _summarize(self._last_event)
                    ),
                    raw=self._last_event.metadata,
                )
                return StepResult(obs=obs, reward=-0.5, done=False, info={"success": False, "event": self._last_event.metadata, "error": "objectId not in metadata"})

            # 3) object should be visible to be interactable (AI2-THOR common constraint)
            # If you want to allow forceAction pickups when not visible, delete this check.
            if not obj_meta.get("visible", False):
                obs = Observation(
                    text=(
                        "PickupObject blocked: target objectId is not currently visible.\n"
                        f"attempted_objectId={obj_id}\n"
                        + _summarize(self._last_event)
                    ),
                    raw=self._last_event.metadata,
                )
                return StepResult(obs=obs, reward=-0.2, done=False, info={"success": False, "event": self._last_event.metadata, "error": "target not visible"})

            # 4) helper to attempt pickup with consistent args
            def _attempt_pickup():
                args = dict(action.args or {})
                # These keys are accepted in many THOR builds; if your build errors, remove them.
                args.setdefault("forceAction", True)
                # args.setdefault("manualInteract", True)  # optional; enable if you want
                return self.controller.step(action="PickupObject", **args)

            # attempt #1
            event = _attempt_pickup()

            # 5) micro-retry if needed (common for countertop objects)
            if not bool(event.metadata.get("lastActionSuccess", False)):
                # small camera/pose adjustments help a lot
                try:
                    self.controller.step(action="LookDown")
                except Exception:
                    pass
                try:
                    self.controller.step(action="RotateRight")
                except Exception:
                    pass

                event = _attempt_pickup()

                # one more scan in the other direction
                if not bool(event.metadata.get("lastActionSuccess", False)):
                    try:
                        self.controller.step(action="RotateLeft")
                        self.controller.step(action="RotateLeft")
                    except Exception:
                        pass
                    event = _attempt_pickup()

            # finalize
            self._last_event = event
            try:
                self.mark_visited(event)
            except Exception:
                pass

            success = bool(event.metadata.get("lastActionSuccess", False))
            err = event.metadata.get("errorMessage", "")

            obs = Observation(text=_summarize(event), raw=event.metadata)

            # keep your existing done/reward shaping (you can generalize later)
            if self.prompt == 0:
                done = _holding_object_type(event, "Apple")  # you can keep this for now
            elif self.prompt == 1:
                done = _holding_object_type(event, "Tomato")  # you can keep this for now
            elif self.prompt == 2:
                done = _holding_object_type(event, "Apple") or _holding_object_type(event, "Tomato") or  _holding_object_type(event, "Toaster") or  _holding_object_type(event, "Vase")
            reward = 10.0 if done else (0.1 if success else -0.2)

            self._save_frame(event, action.tool_name)
            self._step_count += 1
            return StepResult(obs=obs, reward=reward, done=done, info={"success": success, "event": event.metadata, "error": err})


        ###########
        # AI2-THOR allows controller.step(action="MoveAhead", ...) style. :contentReference[oaicite:3]{index=3}
        event = self.controller.step(action=action.tool_name, **(action.args or {}))
        
        self._last_event = event
        self.mark_visited(event)


        success = bool(event.metadata.get("lastActionSuccess", False))
        err = event.metadata.get("errorMessage", "")

        # Example task: "Pick up an apple"
        if self.prompt == 0:
            done = _holding_object_type(event, "Apple")  # you can keep this for now
        elif self.prompt == 1: 
            done = _holding_object_type(event, "Tomato")  # you can keep this for now
        elif self.prompt == 2:                
            done = _holding_object_type(event, "Apple") or _holding_object_type(event, "Tomato") or  _holding_object_type(event, "Toaster") or  _holding_object_type(event, "Vase")
        reward = 10.0 if done else (0.1 if success else -0.2)
        summerize = _summarize(event)
        print(f'Obs summary: {summerize}')
        obs = Observation(text=summerize, raw=event.metadata)
        self._save_frame(event, action.tool_name)
        self._step_count += 1
        return StepResult(obs=obs, reward=reward, done=done, info={"success": success, "event": event.metadata, "error": err})

    # Helper accessors used by the mock proposer/critic
    def last_event(self):
        return self._last_event

    def reachable_positions(self) -> list[dict]:
        return self._reachable_positions

    def mark_visited(self, event) -> None:
        self._visited_pose_keys.add(self._pose_key(event))

    def is_visited(self, event) -> bool:
        return self._pose_key(event) in self._visited_pose_keys

    def _pose_key(self, event) -> str:
        md = event.metadata
        a = md.get("agent", {})
        pos = a.get("position", {})
        rot = a.get("rotation", {})
        hor = a.get("cameraHorizon", 0)
        # quantize a bit so tiny float noise doesn't explode keys
        return f"{pos.get('x',0):.2f},{pos.get('z',0):.2f},{rot.get('y',0):.0f},{hor:.0f}"

    def _find_object_position(self, object_id: str) -> Optional[Dict]:
        """Return the position dict (x,y,z) for an objectId from last_event, or None."""
        if self._last_event is None:
            return None
        for o in self._last_event.metadata.get("objects", []):
            if o.get("objectId") == object_id:
                # many objects include 'position' nested; fallback to parsing id if necessary
                pos = o.get("position") or {
                    "x": o.get("x", None),
                    "y": o.get("y", None),
                    "z": o.get("z", None),
                }
                # ensure keys exist
                if pos and pos.get("x") is not None:
                    return {"x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"])}
        return None


    def _euclidean_sq(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Squared Euclidean distance between two {x,y,z} dicts (useful for ranking)."""
        return (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2


    def _nearest_reachable_to(self, pos: Dict[str, float]) -> Optional[Dict]:
        """Find the nearest reachable (from cached _reachable_positions) to a given pos.
        Returns the reachable position dict or None if no reachable positions cached.
        """
        if not getattr(self, "_reachable_positions", None):
            return None
        best = None
        best_d = float("inf")
        for r in self._reachable_positions:
            d = self._euclidean_sq(pos, r)
            if d < best_d:
                best_d = d
                best = r
        return best
    
    def _k_nearest_reachable_to(self, pos: Dict[str, float], k: int = 15) -> List[Dict]:
        """Return up to k reachable positions nearest to pos (sorted by distance)."""
        reachables = getattr(self, "_reachable_positions", None) or []
        if not reachables:
            return []
        # Sort by squared distance; take top-k
        return sorted(reachables, key=lambda r: self._euclidean_sq(pos, r))[:k]
    

    def _yaw_to_face(self, src: Dict[str, float], dst: Dict[str, float]) -> float:
        """
        Compute yaw (degrees) so the agent at src faces dst.
        AI2-THOR uses y-rotation degrees; 0 ~= +Z, 90 ~= +X (typical).
        """
        dx = dst["x"] - src["x"]
        dz = dst["z"] - src["z"]
        return math.degrees(math.atan2(dx, dz))