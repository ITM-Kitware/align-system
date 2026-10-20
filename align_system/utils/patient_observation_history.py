"""Scenario-local memory of exact, previously visible patient descriptions."""

from copy import deepcopy


PATIENT_HISTORY_GUIDANCE = """PATIENT HISTORY AND CURRENT CARE
patient_history preserves exact descriptions previously observed for each patient
ID, with scene and action provenance. Use it when a current description omits
identity or affiliation information. An omitted field does not retract an
earlier fact. Do not merge different patients who share a name.
Historical descriptions may contain old injuries, vitals, proximity, or claims
that a patient was treated. Use current observations for present clinical status
and the newest explicit observation when facts conflict; state uncertainty if
the conflict cannot be resolved. Last-known tags are recorded separately and
are not a guarantee that a tag is still current.
handled_patients reports the driver's current scene care tracking. A scene
change clears that tracking; historical care does not restore it. History does
not change current action eligibility or inventory. Ground references to past
facts in their provenance rather than claiming they appear in current observations.
"""


class PatientObservationHistory:
    """Keep distinct visible descriptions; tag evidence lives in PatientTagMemory.

    Updated at scenario entry and after successful actions, never by model calls.
    Descriptions are deduplicated exactly and retained across scene transitions.
    A new scenario gets a fresh instance, even when its ID is reused.
    """

    def __init__(self):
        self._descriptions = {}

    def observe(self, state, action_count):
        source = {'scene_id': state.meta_info.scene_id, 'after_action': action_count}
        for patient in state.characters:
            if getattr(patient, 'unseen', False):
                continue
            description = getattr(patient, 'unstructured', None)
            if not patient.id or not isinstance(description, str) or not description.strip():
                continue
            descriptions = self._descriptions.setdefault(patient.id, {})
            if description not in descriptions:
                descriptions[description] = {
                    'value': description, 'first_seen': deepcopy(source),
                }
            descriptions[description]['last_seen'] = deepcopy(source)

    def snapshot(self):
        return deepcopy({patient_id: {'descriptions': list(descriptions.values())}
                         for patient_id, descriptions in sorted(self._descriptions.items())})
