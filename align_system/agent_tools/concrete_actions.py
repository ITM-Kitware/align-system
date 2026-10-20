"""Complete OW assessment candidates from observed patients and inventory.

OWFormatChoicesADMComponent expands templates by patient. Treatment completion
uses the same injury/supply rules as ActionParameterCompletionADMComponent,
including its filtering of parameters already fixed by the server.
"""
from copy import deepcopy
import json

from align_system.algorithms.apply_treatment import treatment_options
from align_system.utils.agent_action_eligibility import (
    CHARACTER_REQUIRED_ACTIONS, TRIAGE_TAGS, eligible_characters)
from align_system.utils.agent_actions import action_identity


def _as_dict(value):
    return value if isinstance(value, dict) else value.to_dict()


def concrete_actions(state, actions, *, treated_patients=(), vitals_checked=(),
                     tagged_patients=(), evac_patients=()):
    """Return detached candidates and explanations for omitted templates.

    Never invent supplies or hidden injuries. A template without enough observed
    information is omitted, not assigned a misleading generic score. Server IDs
    remain unchanged: execute with that ID plus the listed patient/parameters.
    """
    inventory = getattr(state, 'supplies', None)
    supplies = [_as_dict(s) for s in (inventory or [])
                if (_as_dict(s).get('quantity') or 0) > 0]
    stock = {s['type'] for s in supplies}
    candidates, omitted, seen = [], [], set()
    for offered in actions:
        patients = {c.id: c for c in eligible_characters(
            state, offered.action_type, treated_patients=treated_patients,
            vitals_checked=vitals_checked, tagged_patients=tagged_patients,
            evac_patients=evac_patients)}
        if offered.action_type not in CHARACTER_REQUIRED_ACTIONS:
            targets = [None]
        elif offered.character_id is not None:
            targets = [offered.character_id] if offered.character_id in patients else []
        else:
            targets = list(patients)
        if not targets:
            omitted.append(f'{offered.action_id}: no eligible patient for '
                           f'{offered.action_type} at the current location')
        for patient_id in targets:
            params = dict(offered.parameters or {})
            options = [params]
            if offered.action_type == 'TREAT_PATIENT':
                if params.get('treatment') and params.get('location'):
                    options = ([params] if inventory is None or params['treatment'] in stock else [])
                else:
                    injuries = [_as_dict(i) for i in
                                (getattr(patients[patient_id], 'injuries', None) or [])]
                    options = treatment_options(injuries, supplies)['parameters']
                    options = [{**params, **option} for option in options
                               if option['treatment'] in stock
                               and all(not params.get(key) or params[key] == option[key]
                                       for key in ('treatment', 'location'))]
                if not options:
                    omitted.append(f'{offered.action_id} patient={patient_id}: '
                                   'no supported treatment with observed injuries, '
                                   'in-stock supplies, and the fixed parameters')
            elif offered.action_type == 'TAG_CHARACTER' and not params.get('category'):
                options = [{**params, 'category': tag} for tag in TRIAGE_TAGS]
            for option in options:
                candidate = deepcopy(offered)
                if patient_id is not None:
                    candidate.character_id = patient_id
                candidate.parameters = deepcopy(option)
                identity = json.dumps(action_identity(candidate), sort_keys=True)
                if identity not in seen:
                    candidates.append(candidate)
                    seen.add(identity)
    return candidates, omitted
