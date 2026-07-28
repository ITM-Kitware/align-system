import pytest
from copy import deepcopy

from align_system.algorithms.world_state_adm_component import WorldStateTrackerADMComponent
from align_system.algorithms.pipeline_adm import PipelineADM
from align_system.algorithms.random_adm_component import RandomChoiceADMComponent


def test_world_state_tracker_initialization_and_update():
    component = WorldStateTrackerADMComponent()

    scene_1 = {
        "unstructured": "Scene 1: Initial area.",
        "elapsed_time": 0,
        "meta_info": {"scene_id": "scene_1"},
        "supplies": [
            {"type": "Tourniquet", "quantity": 999},
            {"type": "Pressure bandage", "quantity": 999}
        ],
        "characters": [
            {
                "id": "Patient 1",
                "name": "Patient 1",
                "unstructured": "Local civilian with moderate bleeding from their thigh",
                "nearby": True,
                "unseen": False,
                "visited": False,
                "tag": None,
                "vitals": {
                    "avpu": "ALERT",
                    "breathing": "NORMAL",
                    "heart_rate": "FAST"
                }
            },
            {
                "id": "Patient 2",
                "name": "Patient 2",
                "unstructured": "Military member of your unit with a broken left wrist, no bleeding",
                "nearby": True,
                "unseen": False,
                "visited": False,
                "tag": None,
                "vitals": {
                    "avpu": "ALERT",
                    "breathing": "NORMAL",
                    "heart_rate": "FAST"
                }
            },
            {
                "id": "Patient 6",
                "name": "Patient 6",
                "unstructured": "Civilian likely responsible for explosion",
                "nearby": False,
                "unseen": False,
                "visited": False,
                "tag": None
            }
        ]
    }

    # First probe call
    world_state, original_state, stale_info = component.run(scene_1, scenario_id="scenario_1")

    assert world_state["unstructured"] == "Scene 1: Initial area."
    assert len(world_state["characters"]) == 3
    assert world_state["characters"][0]["vitals"]["avpu"] == "ALERT"
    assert original_state == scene_1
    assert stale_info["stale_characters"] == {}

    # Second probe call:
    # - Move away from Patient 1 (nearby becomes False, vitals omitted)
    # - Patient 6 becomes nearby with vitals assessed
    # - Tourniquet used (quantity decreases from 999 to 998)
    scene_2 = {
        "unstructured": "Scene 2: Moved to patient 6 area.",
        "elapsed_time": 10,
        "meta_info": {"scene_id": "scene_2"},
        "supplies": [
            {"type": "Tourniquet", "quantity": 998},
            {"type": "Pressure bandage", "quantity": 999}
        ],
        "characters": [
            {
                "id": "Patient 1",
                "name": "Patient 1",
                "unstructured": "Patient 1 far away.",
                "nearby": False,
                "unseen": False,
                "visited": True,  # dynamic field updated
                "tag": "RED"
                # vitals omitted in scene_2
            },
            {
                "id": "Patient 2",
                "name": "Patient 2",
                "unstructured": "Military member of your unit with a broken left wrist, no bleeding",
                "nearby": True,
                "unseen": False,
                "visited": True,
                "tag": None,
                "vitals": {
                    "avpu": "ALERT",
                    "breathing": "NORMAL",
                    "heart_rate": "FAST"
                }
            },
            {
                "id": "Patient 6",
                "name": "Patient 6",
                "unstructured": "Civilian likely responsible for the explosion with moderate bleeding from their torso; possible internal bleeding",
                "nearby": True,
                "unseen": False,
                "visited": True,
                "tag": None,
                "vitals": {
                    "avpu": "VOICE",
                    "breathing": "FAST",
                    "heart_rate": "FAST"
                }
            }
        ]
    }

    world_state_2, original_state_2, stale_info_2 = component.run(scene_2, scenario_id="scenario_1")

    # Patient 1 should RETAIN detailed unstructured description AND vitals from scene 1
    p1 = next(c for c in world_state_2["characters"] if c["id"] == "Patient 1")
    assert p1["unstructured"] == "Local civilian with moderate bleeding from their thigh"
    assert p1["vitals"]["avpu"] == "ALERT"
    assert p1["vitals"]["breathing"] == "NORMAL"
    assert p1["visited"] is True
    assert p1["tag"] == "RED"
    assert p1["nearby"] is False

    # Patient 6 was nearby=False in scene 1, nearby=True in scene 2 -> Should update with new detailed description and vitals
    p6 = next(c for c in world_state_2["characters"] if c["id"] == "Patient 6")
    assert p6["unstructured"] == "Civilian likely responsible for the explosion with moderate bleeding from their torso; possible internal bleeding"
    assert p6["vitals"]["avpu"] == "VOICE"
    assert p6["nearby"] is True

    # Supplies quantity should be updated to 998
    t_supply = next(s for s in world_state_2["supplies"] if s["type"] == "Tourniquet")
    assert t_supply["quantity"] == 998

    # Stale info should track Patient 1 as having stale fields
    assert "Patient 1" in stale_info_2["stale_characters"]
    assert "unstructured" in stale_info_2["stale_characters"]["Patient 1"]["stale_fields"]
    assert "vitals" in stale_info_2["stale_characters"]["Patient 1"]["stale_fields"]


def test_vitals_recursive_accumulation():
    component = WorldStateTrackerADMComponent()

    scene_1 = {
        "characters": [
            {
                "id": "P1",
                "nearby": True,
                "vitals": {"avpu": "ALERT"}
            }
        ]
    }

    # Step 1: Initial vitals (avpu)
    ws1, _, _ = component.run(scene_1, scenario_id="s1")
    assert ws1["characters"][0]["vitals"] == {"avpu": "ALERT"}

    # Step 2: Assessment adds spo2
    scene_2 = {
        "characters": [
            {
                "id": "P1",
                "nearby": True,
                "vitals": {"spo2": "98%"}
            }
        ]
    }
    ws2, _, _ = component.run(scene_2, scenario_id="s1")
    # Both avpu and spo2 should be present
    assert ws2["characters"][0]["vitals"]["avpu"] == "ALERT"
    assert ws2["characters"][0]["vitals"]["spo2"] == "98%"


def test_world_state_tracker_reset_history_and_scenario_change():
    component = WorldStateTrackerADMComponent()

    scene_s1 = {
        "unstructured": "Scenario 1 Scene",
        "characters": [{"id": "P1", "unstructured": "Details S1", "nearby": True}]
    }

    component.run(scene_s1, scenario_id="scenario_1")
    assert component.world_state is not None

    # Reset history
    component.reset_history()
    assert component.world_state is None
    assert component.current_scenario_id is None

    # Auto reset on scenario ID change
    component.run(scene_s1, scenario_id="scenario_1")
    assert component.world_state["unstructured"] == "Scenario 1 Scene"

    scene_s2 = {
        "unstructured": "Scenario 2 Scene",
        "characters": [{"id": "P10", "unstructured": "Details S2", "nearby": True}]
    }

    world_state_s2, _, _ = component.run(scene_s2, scenario_id="scenario_2")
    assert world_state_s2["unstructured"] == "Scenario 2 Scene"
    assert len(world_state_s2["characters"]) == 1
    assert world_state_s2["characters"][0]["id"] == "P10"


def test_pipeline_adm_reset_history_propagation():
    tracker = WorldStateTrackerADMComponent()
    random_comp = RandomChoiceADMComponent()

    pipeline = PipelineADM(steps=[tracker, random_comp])

    scene = {"unstructured": "Test", "characters": []}
    tracker.run(scene, scenario_id="test_scen")

    assert tracker.world_state is not None

    # Call reset_history on PipelineADM
    pipeline.reset_history()

    assert tracker.world_state is None
