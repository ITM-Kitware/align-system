from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class _OpenWorldBase(BaseModel):
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class OpenWorldMetaInfo(_OpenWorldBase):
    scene_id: str


class OpenWorldCharacter(_OpenWorldBase):
    id: str
    name: str
    unstructured: str
    intent: Optional[str] = None
    rapport: Optional[str] = None
    unseen: Optional[bool] = False


class OpenWorldAction(_OpenWorldBase):
    action_id: str
    action_type: Optional[str] = None
    unstructured: str
    character_id: Optional[str] = None
    intent_action: bool = False
    kdma_association: Optional[Any] = None
    justification: Optional[str] = None


class OpenWorldState(_OpenWorldBase):
    unstructured: str
    elapsed_time: int = 0
    scenario_complete: bool
    meta_info: OpenWorldMetaInfo
    characters: List[OpenWorldCharacter]


def hydrate_scenario_state(record):
    """ Hydrate scenario state from p1 record """
    from align_system.data_models.compat.ta3_ph1_client_models import (
        State,
        Action,
        Character,
        Supplies,
        Injury,
        Environment,
        DecisionEnvironment,
        Aid,
        SimEnvironment,
        MetaInfo,
    )

    state = State(**record['full_state'])
    state.meta_info = MetaInfo(**state.meta_info)
    # For some reason this initialization from a dictionary
    # doesn't recursively init; need to manually do it
    state.characters = [Character(**c) for c in state.characters]
    for c in state.characters:
        c.injuries = [Injury(**i) for i in c.injuries]
    state.supplies = [Supplies(**s) for s in state.supplies]
    state.environment = Environment(**state.environment)
    state.environment.decision_environment = DecisionEnvironment(
        **state.environment.decision_environment)
    if state.environment.decision_environment.aid is not None:
        state.environment.decision_environment.aid = [
            Aid(**a) for a in state.environment.decision_environment.aid]
    state.environment.sim_environment = SimEnvironment(
        **state.environment.sim_environment)

    actions = [Action(**a) for a in record['choices']]
    # TODO: Fix this on the input-output generation side, need
    # to make sure original choices aren't being modified by
    # ADM; for now manually clearing the justification string
    for a in actions:
        a.justification = None

    return state, actions

def p2triage_hydrate_scenario_state(record):
    """ Hydrate scenario state from p2triage record """
    from swagger_client.models import (
        State,
        Action,
    )

    state = State(**record['full_state'])

    actions = [Action(**a) for a in record['choices']]
    # TODO: Fix this on the input-output generation side, need
    # to make sure original choices aren't being modified by
    # ADM; for now manually clearing the justification string
    for a in actions:
        a.justification = None

    return state, actions

def open_world_hydrate_scenario_state(record):
    """Hydrate open-world scenario state; no environment/supplies required."""
    full_state = record['full_state']

    state = OpenWorldState(
        unstructured=full_state['unstructured'],
        elapsed_time=full_state.get('elapsed_time', 0),
        scenario_complete=full_state['scenario_complete'],
        meta_info=OpenWorldMetaInfo(**full_state['meta_info']),
        characters=[OpenWorldCharacter(**c) for c in full_state.get('characters', [])],
    )

    actions = [
        OpenWorldAction(**a, justification=None)
        for a in record['choices']
    ]

    return state, actions


def minimal_hydrate_scenario_state(record):
    """ Hydrate scenario state from minimal record """
    from collections import namedtuple

    full_state = record['full_state']
    MetaInfo = namedtuple('MetaInfo', ['scene_id'])
    State = namedtuple('State', ['unstructured', 'scenario_complete', 'meta_info'])
    meta_info = MetaInfo(full_state['meta_info']['scene_id'])
    state = State(full_state['unstructured'], full_state['scenario_complete'], meta_info)

    Action = namedtuple('Action', ['action_id', 'unstructured', 'justification', 'kdma_association'])
    actions = [
        Action(
            a['action_id'],
            a["unstructured"],
            a['justification'] if 'justification' in a else None,
            a['kdma_association'] if 'kdma_association' in a else None
        )
        for a in record['choices']
    ]

    return state, actions
