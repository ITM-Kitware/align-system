"""Patient eligibility shared by OW execution and value assessments."""
from swagger_client.models import ActionTypeEnum

from align_system.data_models.compat.ta3_ph1_client_models import CharacterTagEnum
from align_system.utils import get_swagger_class_enum_values


TRIAGE_TAGS = get_swagger_class_enum_values(CharacterTagEnum)
CHARACTER_REQUIRED_ACTIONS = frozenset({
    ActionTypeEnum.CHECK_VITALS,
    ActionTypeEnum.TREAT_PATIENT,
    ActionTypeEnum.MOVE_TO,
    ActionTypeEnum.MOVE_TO_EVAC,
    ActionTypeEnum.TAG_CHARACTER,
})
_NEARBY_ACTIONS = frozenset({
    ActionTypeEnum.CHECK_VITALS,
    ActionTypeEnum.TREAT_PATIENT,
    ActionTypeEnum.TAG_CHARACTER,
})


def eligible_characters(state, action_type, *, treated_patients=(),
                        vitals_checked=(), tagged_patients=(), evac_patients=()):
    """Return server-listed patients who can receive this action.

    Unknown proximity preserves the server's older observation contract.
    MOVE_TO and MOVE_TO_EVAC may use an unseen patient's listed identity,
    without treating hidden clinical fields as observed evidence. Evacuation
    has no visibility or proximity restriction: OW can offer evacuation alone
    while patients at other locations remain unseen. Callers must still honor
    the server's offered action templates and any fixed patient targets.
    The caller supplies any per-scene duplicate-action memory it wants applied.
    """
    if action_type not in CHARACTER_REQUIRED_ACTIONS:
        return []
    excluded = set(evac_patients)
    if action_type == ActionTypeEnum.TREAT_PATIENT:
        excluded.update(treated_patients)
    elif action_type == ActionTypeEnum.CHECK_VITALS:
        excluded.update(vitals_checked)
    elif action_type == ActionTypeEnum.TAG_CHARACTER:
        excluded.update(tagged_patients)

    eligible = []
    for patient in getattr(state, 'characters', None) or []:
        if getattr(patient, 'id', None) is None or patient.id in excluded:
            continue
        nearby = getattr(patient, 'nearby', None)
        if action_type == ActionTypeEnum.MOVE_TO:
            if nearby is True:
                continue
        elif (action_type != ActionTypeEnum.MOVE_TO_EVAC
              and getattr(patient, 'unseen', False)):
            continue
        if action_type in _NEARBY_ACTIONS and nearby is False:
            continue
        if (action_type == ActionTypeEnum.TAG_CHARACTER
                and getattr(patient, 'tag', None) is not None):
            continue
        eligible.append(patient)
    return eligible
