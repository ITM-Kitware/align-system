"""Scenario-local memory of exact, previously visible patient observations.

The ledger consumes the driver's sanitized observation view, never raw state or
evaluation labels. It preserves evidence; it does not extract patient attributes
or promote historical clinical information into the current observation.
"""

from collections.abc import Mapping
from copy import deepcopy
from enum import Enum


PATIENT_HISTORY_GUIDANCE = """\
patient_history contains exact descriptions previously observed for the same
patient IDs and the last known triage tag, with scene and action provenance.
Use that history when a current description omits previously observed identity
or affiliation information, or when a previously observed tag is omitted. A
shorter description or an omitted field does not retract an earlier fact.
Do not confuse different patients with the same name or merge their histories.

Historical descriptions can also contain old injuries, vitals, or proximity.
They are historical evidence, not a claim that those conditions remain current.
Use current observations for present clinical status and the newest explicit
observation when facts conflict; state uncertainty if the conflict cannot be
resolved. last_known_tag records a past observation or successful tag assignment,
not a guarantee that the tag is still current. History does not change current
action eligibility, inventory, or scene-specific handled-patient bookkeeping.
Ground references to earlier facts in patient_history and its provenance rather
than claiming that omitted information appears in the current observation.
"""


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(
        value, name, default)


def _text(value):
    if isinstance(value, Enum):
        value = value.value
    return value if isinstance(value, str) and value.strip() else None


class PatientObservationHistory:
    """Preserve descriptions and last known tags across one scenario's scenes.

    Call ``update`` with the initial sanitized observation and after every
    successful action's returned state. ``action_index`` for observations is the
    next action index (zero for the initial state), matching the visible context.
    Repeated model/tool calls need not update the ledger.

    Call ``record_tag`` only after a successful environment submission, before
    ingesting its returned observation. Its index identifies the executed action,
    so an initial tag action has index zero and its returned observation index
    one. An explicit returned tag takes precedence over the assignment event.

    Descriptions are kept in first-seen order and deduplicated exactly. Growth
    depends on distinct observed descriptions, not model calls. No observations
    are silently truncated or evicted; the driver's mandatory context budget
    decides whether the complete history can be presented. Create a fresh
    instance for each scenario, including repeated evaluations of the same ID.
    """

    def __init__(self):
        self._patients = {}
        self._description_indices = {}

    def _patient(self, patient_id):
        if patient_id not in self._patients:
            self._patients[patient_id] = {
                'descriptions': [],
                'last_known_tag': None,
            }
            self._description_indices[patient_id] = {}
        return self._patients[patient_id]

    def update(self, observation: Mapping, action_index: int):
        """Ingest only visible IDs, exact descriptions, and nonempty tags."""
        source = {
            'scene_id': observation['scene_id'],
            'action_index': action_index,
        }
        for casualty in observation.get('casualties', []):
            if casualty.get('unseen', False):
                continue
            patient_id = _text(casualty.get('id'))
            if patient_id is None:
                continue
            patient = self._patient(patient_id)
            description = _text(casualty.get('unstructured'))
            if description is not None:
                indices = self._description_indices[patient_id]
                if description not in indices:
                    indices[description] = len(patient['descriptions'])
                    patient['descriptions'].append({
                        'value': description,
                        'first_seen': deepcopy(source),
                        'last_seen': deepcopy(source),
                    })
                else:
                    patient['descriptions'][indices[description]][
                        'last_seen'] = deepcopy(source)
            tag = _text(casualty.get('tag'))
            if tag is not None:
                patient['last_known_tag'] = {
                    'value': tag,
                    'source': 'observation',
                    **deepcopy(source),
                }

    def record_tag(self, action, scene_id: str, action_index: int):
        """Record a successful actual tag assignment, never an intent action.

        This method cannot determine submission success; the session must call
        it only after the environment has accepted the action.
        """
        if (_text(_field(action, 'action_type')) != 'TAG_CHARACTER'
                or _field(action, 'intent_action', False)):
            return
        patient_id = _text(_field(action, 'character_id'))
        parameters = _field(action, 'parameters')
        tag = _text(_field(parameters, 'category'))
        if patient_id not in self._patients or tag is None:
            return
        self._patients[patient_id]['last_known_tag'] = {
            'value': tag,
            'source': 'successful_tag_action',
            'scene_id': scene_id,
            'action_index': action_index,
        }

    def snapshot(self) -> dict:
        """Return independent JSON-safe evidence with stable patient ordering."""
        return deepcopy(dict(sorted(self._patients.items())))
