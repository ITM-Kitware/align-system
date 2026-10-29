"""Observed patient choices for Kaleido; never execute or infer hidden facts."""
import json
import re

from align_system.utils.agent_action_eligibility import eligible_characters
from align_system.utils.patient_observation_history import PATIENT_HISTORY_GUIDANCE


DECISION_ACTIONS = {
    'treat': {'TREAT_PATIENT', 'CHECK_VITALS', 'MOVE_TO'},
    'evacuate': {'MOVE_TO_EVAC'},
}


def _observed_fields(patient):
    fields = ('id', 'name', 'unseen', 'nearby')
    if not getattr(patient, 'unseen', False):
        fields += ('unstructured', 'demographics', 'injuries', 'vitals', 'tag')

    def clean(value):
        if hasattr(value, 'to_dict'):
            value = value.to_dict()
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if v is not None and k != 'kdma_association'}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return {key: clean(getattr(patient, key)) for key in fields
            if getattr(patient, key, None) is not None}


def patient_choices(session, actions, decision):
    """Build one row per patient supported by an offered action.

    'treat' means prioritizing care, including moving to assess a visible
    casualty. It does not promise that a particular treatment is executable.
    Evacuation may select a server-listed unseen casualty, with unknown current
    clinical details; previously observed descriptions remain labeled history.
    """
    if decision not in DECISION_ACTIONS:
        raise ValueError("decision must be 'treat' or 'evacuate'")
    state = session.current_state
    routes = {}
    evacuated = getattr(session, 'evac_patients', ())
    for action in actions:
        if action.action_type not in DECISION_ACTIONS[decision]:
            continue
        for patient in eligible_characters(
                state, action.action_type, evac_patients=evacuated):
            if decision == 'treat' and getattr(patient, 'unseen', False):
                continue
            if action.character_id is not None and action.character_id != patient.id:
                continue
            route = {'action_id': action.action_id, 'action_type': action.action_type,
                     'parameters': action.parameters or {}}
            if route not in routes.setdefault(patient.id, []):
                routes[patient.id].append(route)

    history = getattr(session, 'patient_history', None)
    history = history.snapshot() if history is not None else {}
    rows = []
    for patient in state.characters:
        if patient.id not in routes:
            continue
        rows.append({
            'patient_id': patient.id,
            'observed': _observed_fields(patient),
            'history': history.get(patient.id, {}),
            'care_this_scene': {
                key: patient.id in getattr(session, attr, ())
                for key, attr in (
                    ('treated', 'treated_patients'), ('tagged', 'tagged_patients'),
                    ('vitals_checked', 'vitals_checked'))
            },
            'offered_actions': routes[patient.id],
        })
    return rows


def _fact_text(value):
    """Render observed structured fields without JSON syntax or null entries."""
    if isinstance(value, dict):
        return '; '.join(f"{key.replace('_', ' ')}: {text}"
                         for key, item in value.items()
                         if (text := _fact_text(item)))
    if isinstance(value, list):
        return '; '.join(text for item in value if (text := _fact_text(item)))
    if value is None:
        return ''
    return ' '.join(str(value).split())


def _description_parts(text):
    # Keep the source wording (including uncertainty). This is exact-text
    # deduplication, not a model-generated or semantic summary.
    return re.split(r'(?<=[.!?])\s+', ' '.join((text or '').split()))


def _compact_patient(row):
    observed = row['observed']
    lines = [str(row['patient_id'])]
    for field, label in (
            ('name', 'Name'), ('unstructured', 'Description'),
            ('demographics', 'Demographics'), ('injuries', 'Injuries'),
            ('vitals', 'Vitals'), ('tag', 'Tag')):
        text = _fact_text(observed.get(field))
        if text:
            lines.append(f'{label}: {text}')
    nearby = observed.get('nearby')
    lines.append('Proximity: ' + ('nearby' if nearby is True else
                                 'distant' if nearby is False else 'unknown'))
    if observed.get('unseen'):
        lines.append('Currently unseen; current clinical details unknown.')
    elif not _fact_text(observed.get('vitals')):
        lines.append('Vitals: unknown')
    care = row['care_this_scene']
    lines.append('Care this scene: ' + '; '.join(
        f"{key.replace('_', ' ')}: {'recorded' if value else 'not recorded'}"
        for key, value in care.items()))

    # Newest observations first; omit sentences already present in the current
    # description or a more recent history entry. Keep other wording verbatim
    # and labeled as potentially stale rather than guessing which facts matter.
    seen = set(_description_parts(observed.get('unstructured')))
    history = sorted(row['history'].get('descriptions', []),
                     key=lambda entry: entry.get('last_seen', {}).get('after_action', -1),
                     reverse=True)
    for entry in history:
        missing = []
        for part in _description_parts(entry.get('value')):
            if part and part not in seen:
                missing.append(part)
                seen.add(part)
        if missing:
            lines.append('Historical (may be stale): ' + ' '.join(missing))
    return '\n'.join(lines)


def _compact_patient_query(state, rows, index, decision):
    choice = ('evacuation' if decision == 'evacuate' else 'care')
    alternatives = '\n\n'.join(_compact_patient(row)
                                 for i, row in enumerate(rows) if i != index)
    return (
        f"Action: Prioritize patient {rows[index]['patient_id']} for {choice} next "
        "ahead of the other eligible patients.\n\n"
        f"Situation: {state.unstructured or 'unknown'}\n\n"
        f"Selected patient:\n{_compact_patient(rows[index])}\n\n"
        f"Other eligible patients:\n{alternatives or 'None'}\n\n"
        "Care priority includes moving or assessing first if needed. "
        "Evaluate the value expressed by this choice, not the patient's worth "
        "or clinical urgency. Missing facts remain unknown; do not infer "
        "affiliation from names. Current observations take precedence over "
        "history; history is listed newest first and does not establish current "
        "care status or action eligibility.")


def patient_query(state, rows, index, decision, prompt_format='full'):
    """Put the selected patient first; include alternatives as factual context."""
    if prompt_format == 'compact':
        return _compact_patient_query(state, rows, index, decision)
    if prompt_format != 'full':
        raise ValueError("patient_prompt_format must be 'full' or 'compact'")
    selected = rows[index]
    alternatives = [row for i, row in enumerate(rows) if i != index]
    choice = ('evacuate next' if decision == 'evacuate'
              else 'prioritize for care next, moving or assessing first if needed')
    return (
        f"Choosing patient {selected['patient_id']} to {choice} "
        "ahead of the other listed patients. Assess the value expressed by this "
        "priority decision, not the worth of the person or a clinical urgency score.\n"
        f"Selected patient: {json.dumps(selected, sort_keys=True, default=str)}\n"
        f"Other patient choices: {json.dumps(alternatives, sort_keys=True, default=str)}\n"
        f"Scene: {state.unstructured}\n"
        "Missing facts are unknown. Do not infer affiliation from names. "
        "Offered actions constrain the choice; this assessment authorizes no action.\n"
        + PATIENT_HISTORY_GUIDANCE)


def format_patient_assessment(decision, targets, rows, unscored):
    lines = [
        f"Patient priority assessment: {decision}",
        "Alignment target: " + ", ".join(f"{k}={v:.2f}" for k, v in targets.items()),
        "Scores describe value relevance/support for prioritizing each patient. "
        "They are not clinical urgency scores, calibrated TA1 scores, or a patient "
        "ranking. Interpret the target using the driver's alignment guidance; "
        "a target of zero does not by itself mean favoring the opposite group.",
    ]
    if unscored:
        lines.append("Unscored attributes (no description): " + ", ".join(unscored))
    for row in rows:
        lines.append(f"- {row['patient_id']}:")
        lines.append("    Observed: " + json.dumps(row['observed'], default=str))
        if row['history']:
            lines.append("    Previously observed: " + json.dumps(row['history'], default=str))
        lines.append("    Care this scene: " + json.dumps(row['care_this_scene']))
        for attribute, score in row['scores'].items():
            lines.append(
                f"    {attribute}: relevance {score['relevant']:.2f}; "
                f"supports {score['supports']:.2f} / opposes {score['opposes']:.2f} / "
                f"either {score['either']:.2f}; support estimate {score['estimate']:.2f}")
            if score.get('explanation'):
                lines.append(f"    Model explanation: {score['explanation']}")
    lines.append("Choose an available action with the patient's exact ID and valid "
                 "parameters; use current clinical observations and labeled history.")
    return "\n".join(lines)
