"""Remember triage tags learned from observations and successful actions."""
from copy import deepcopy
import json


class PatientTagMemory:
    """Last-known tags, kept separate from the environment's current state."""

    def __init__(self):
        self._tags = {}

    def _remember(self, patient_id, patient_name, tag, scene_id, source, action_count):
        if not patient_id or not tag:
            return
        previous = self._tags.get(patient_id)
        # Repeated observations need not replace the provenance of a tag we
        # applied. A new value or scene is fresh evidence and replaces it.
        if (source == 'observation' and previous
                and previous['tag'] == tag and previous['scene_id'] == scene_id):
            return
        self._tags[patient_id] = {
            'patient_id': patient_id, 'patient_name': patient_name or patient_id,
            'tag': tag, 'scene_id': scene_id, 'source': source,
            'after_action': action_count,
        }

    def observe(self, state, action_count):
        for patient in state.characters:
            if not getattr(patient, 'unseen', False):
                self._remember(
                    patient.id, patient.name, getattr(patient, 'tag', None),
                    state.meta_info.scene_id, 'observation', action_count)

    def record_success(self, action, state, action_count):
        if (action.action_type != 'TAG_CHARACTER'
                or getattr(action, 'intent_action', False)):
            return
        patient = next((p for p in state.characters
                        if p.id == action.character_id), None)
        self._remember(
            action.character_id, getattr(patient, 'name', None),
            (action.parameters or {}).get('category'), state.meta_info.scene_id,
            'successful_tag_action', action_count)

    def snapshot(self):
        return deepcopy(list(self._tags.values()))

    def tagged_in_scene(self, scene_id):
        """Patient IDs with a successful or observed tag in this scene."""
        return {patient_id for patient_id, entry in self._tags.items()
                if entry['scene_id'] == scene_id}

    def context(self, state):
        if not self._tags:
            return ''
        observed = {p.id: getattr(p, 'tag', None) for p in state.characters
                    if not getattr(p, 'unseen', False)}
        entries = [{**entry,
                    'same_scene': entry['scene_id'] == state.meta_info.scene_id,
                    'currently_observed': observed.get(entry['patient_id']) == entry['tag']}
                   for entry in self._tags.values()]
        return (
            'LAST-KNOWN PATIENT TAGS\n'
            'These are records of your successful tagging actions and tags you '
            'actually observed. A tag missing from the current observation means '
            'unknown, not untagged. Do not repeatedly move to or retag a patient '
            'solely because their tag is temporarily absent. Records from an '
            'earlier scene are historical and require confirmation in this scene. '
            'Use new observations when they contradict older records.\n'
            + json.dumps(entries, ensure_ascii=False))
