"""Driver care-credit history and optional navigation reassessment.

This ledger explains the driver's existing completion rules. It does not infer
medical recovery from prose, invent observations, or change those rules.
"""
from copy import deepcopy
import json


RESET_AWARE_GUIDANCE = """CURRENT CARE CREDIT
Use patient_progress to distinguish prior treatment, current treatment credit,
and recorded triage tags. The driver resets treatment credit on a scene change
and can reopen care after newly observed clinical changes. A description saying
'treated' can describe earlier care: it does not establish current credit.
When treatment_credit_reset is present and currently_treated is false, prior
care no longer counts toward scene completion. Reconsider that patient and
perform a fresh treatment when the environment offers it. A recorded tag does
not satisfy treatment. This qualifies the instruction to move on after care:
move on only when current care is complete. Use currently offered actions;
an evacuation-only scene does not require another treatment or move_to."""


class PatientProgress:
    def __init__(self):
        self.treatments = {}
        self.resets = {}
        self.facts = {}
        self.visits = {}
        self.reassessments = {}

    def update(self, session, action, previous_scene, previous_treated, clinical):
        """Called after successful execution and the existing credit update."""
        scene = session.current_state.meta_info.scene_id
        patient_id = action.character_id
        if action.action_type == 'TREAT_PATIENT' and not getattr(action, 'intent_action', False):
            self.treatments[patient_id] = {
                'scene_id': previous_scene, 'after_action': session.n_actions}
            previous_treated = {*previous_treated, patient_id}
        changed_scene = scene != previous_scene
        if changed_scene:
            self.facts.clear()
            self.visits.clear()
            self.reassessments.clear()
        for patient_id in previous_treated - session.treated_patients:
            self.resets[patient_id] = {
                'reason': 'scene_change' if changed_scene else 'clinical_change',
                'previous_scene_id': previous_scene, 'scene_id': scene,
                'after_action': session.n_actions}
        for patient_id in session.treated_patients:
            self.resets.pop(patient_id, None)
        for patient_id, facts in clinical.items():
            self.facts.setdefault(patient_id, {}).update(deepcopy(facts))
        if action.action_type == 'MOVE_TO' and not changed_scene:
            self.visits[action.character_id] = self.signature(session, action.character_id)
            self.reassessments.pop(action.character_id, None)

    def snapshot(self, session):
        session.pending_patients()  # Reconcile freshly observed tags first.
        tags = {entry['patient_id']: entry for entry in session.tag_memory.snapshot()}
        return [{
            'patient_id': patient.id, 'patient_name': patient.name,
            'previously_treated': patient.id in self.treatments,
            'last_treatment': deepcopy(self.treatments.get(patient.id)),
            'currently_treated': patient.id in session.treated_patients,
            'treatment_credit_reset': deepcopy(self.resets.get(patient.id)),
            'tagged_current_scene': patient.id in session.tagged_patients,
            'last_known_tag': deepcopy(tags.get(patient.id)),
        } for patient in session.current_state.characters]

    @staticmethod
    def pending_actions(session, patient_id):
        """Required care offered for a patient, irrespective of distance."""
        if patient_id in session.evac_patients:
            return []
        done = {
            'TREAT_PATIENT': patient_id in session.treated_patients,
            'TAG_CHARACTER': patient_id in session.tagged_patients,
            'CHECK_VITALS': patient_id in session.vitals_checked,
        }
        if done['TREAT_PATIENT'] and done['TAG_CHARACTER']:
            return []
        return sorted({action.action_type for action in session.available_actions
                       if action.action_type in done and not done[action.action_type]
                       and action.character_id in (None, patient_id)})

    def signature(self, session, patient_id):
        # Position/visibility changes alone cannot make A -> B -> A progress.
        tags = {entry['patient_id']: entry['tag']
                for entry in session.tag_memory.snapshot()}
        return json.dumps([
            patient_id in session.treated_patients,
            patient_id in session.tagged_patients,
            patient_id in session.vitals_checked,
            tags.get(patient_id), self.facts.get(patient_id, {}),
        ], sort_keys=True, default=str)

    def navigation_error(self, session, patient_id):
        # Movement outside care scenes retains the environment's own rules.
        if not any(a.action_type in {'TREAT_PATIENT', 'TAG_CHARACTER', 'CHECK_VITALS'}
                   for a in session.available_actions):
            return None
        if not self.pending_actions(session, patient_id):
            return (f'{patient_id} has no pending offered care action. '
                    'Reassess the current patient progress and choose a patient '
                    'who needs an action before moving.')
        signature = self.signature(session, patient_id)
        if (self.visits.get(patient_id) == signature
                and self.reassessments.get(patient_id) != signature):
            return (f'Revisiting {patient_id} would repeat a visit without changed '
                    'care or disclosed clinical facts. Reassess your plan using '
                    'reassess_navigation with a pending next_action and justification '
                    'before moving there again.')
        return None

    def reassess(self, session, patient_id, next_action, justification):
        next_action = next_action.upper()
        if not justification.strip() or next_action not in self.pending_actions(session, patient_id):
            return ('Reassessment requires a justification and a pending offered '
                    'next_action: ' + ', '.join(self.pending_actions(session, patient_id)))
        self.reassessments[patient_id] = self.signature(session, patient_id)
        session.record_event('navigation_reassessed', decision_index=session.n_actions,
                             patient_id=patient_id, next_action=next_action,
                             justification=justification)
        return (f'Plan reassessed: move to {patient_id}, then {next_action}. '
                'Review the new observation on arrival.')

    def context(self, session):
        return RESET_AWARE_GUIDANCE + '\n' + json.dumps(self.snapshot(session), ensure_ascii=False)
