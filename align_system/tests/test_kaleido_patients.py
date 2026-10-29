"""Patient evidence, eligibility, caching, and driver integration without a GPU."""
from copy import deepcopy
import json
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from pathlib import Path
import pytest

from align_system.agent_tools.kaleido import KaleidoToolkit
from align_system.agent_tools.kaleido_patients import patient_choices, patient_query
from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver
from align_system.tests.test_agent_assessments import Model, action, session_for


class Scorer:
    def __init__(self):
        self.calls = []
        self.explanations = 0

    def get_relevance(self, texts, *args, **kwargs):
        self.calls.append(list(texts))
        return [[0.9, 0.1] for _ in texts]

    def get_valence(self, texts, *args, **kwargs):
        return [[0.7, 0.1, 0.2] for _ in texts]

    def get_explanation(self, texts, *args, **kwargs):
        self.explanations += 1
        return ['Value evidence from the observed patient description.' for _ in texts]


def toolkit(**kwargs):
    return KaleidoToolkit(
        Scorer(), {'affiliation': {'description': 'Favoring your own group'}},
        **kwargs)


def patient(identifier, description, **kwargs):
    return Model(**{
        'id': identifier, 'name': 'Same name', 'unstructured': description,
        'unseen': False, 'nearby': True, 'tag': 'IMMEDIATE',
        'injuries': [{'name': 'Laceration', 'location': 'left thigh'}],
        'vitals': {'mental_status': 'ALERT'},
        'demographics': {'military_disposition': 'Civilian'},
        'kdma_association': {'affiliation': 'HIDDEN_GROUND_TRUTH'},
        **kwargs,
    })


def evacuation_session():
    session, records = session_for([action('evac', 'MOVE_TO_EVAC')])
    session.current_state.characters = [
        patient('p1', 'Soldier in your unit, leg wound.'),
        patient('p2', 'Local civilian, leg wound.'),
    ]
    session.patient_history.observe(session.current_state, action_count=0)
    return session, records


def test_distinct_patient_choices_for_one_generic_evacuation_template():
    session, records = evacuation_session()
    before = deepcopy(session.current_state.to_dict())
    provider = toolkit()
    tools = {t.name: t for t in provider.build_tools(session)}
    output = tools['assess_patients'].invoke({'decision': 'evacuate'})

    assert '- p1:' in output and '- p2:' in output
    assert 'Model explanation:' in output
    assert 'Closest to target' not in output and 'distance' not in output
    assert len(provider.kaleido.calls[0]) == 2
    first, second = provider.kaleido.calls[0]
    assert first.startswith('Choosing patient p1')
    assert second.startswith('Choosing patient p2')
    assert all('Soldier in your unit' in t and 'Local civilian' in t
               for t in (first, second))
    assert 'HIDDEN_GROUND_TRUTH' not in first + second + output
    assert session.current_state.to_dict() == before
    assert not session.scenario.executed and not records
    assert session.n_actions == 0
    evidence = session.assessments['kaleido_patients']
    assert evidence['decision'] == 'evacuate'
    assert [r['patient_id'] for r in evidence['rows']] == ['p1', 'p2']
    assert evidence['rows'][0]['scores']['affiliation']['estimate'] == pytest.approx(.8)
    assert 'distance' not in evidence['rows'][0]['scores']['affiliation']
    assert provider.kaleido.explanations == 1


def test_automatic_patient_assessment_does_not_skip_single_action_template():
    session, _ = evacuation_session()
    provider = toolkit(auto_assess=True)
    provider.build_tools(session)
    assert 'Patient priority assessment: evacuate' in provider.decision_context(session)
    assert len(provider.kaleido.calls[0]) == 2


def test_unseen_patient_uses_only_identity_and_labeled_prior_observations():
    session, _ = evacuation_session()
    session.current_state.characters[0].unstructured = 'New hidden affiliation SECRET'
    session.current_state.characters[0].unseen = True
    session.current_state.characters[0].vitals = {'mental_status': 'SECRET'}
    provider = toolkit()
    output = provider.patient_assessment_text(session, 'evacuate')
    query = provider.kaleido.calls[0][0]
    assert 'SECRET' not in query + output
    assert 'Soldier in your unit, leg wound.' in query
    assert 'first_seen' in query and 'Previously observed:' in output
    assert 'Historical descriptions may contain old injuries' in query


def test_fixed_server_patient_and_already_evacuated_constraints():
    session, _ = evacuation_session()
    actions = [action('evac', 'MOVE_TO_EVAC', character_id='p2')]
    assert [r['patient_id'] for r in patient_choices(session, actions, 'evacuate')] == ['p2']
    session.evac_patients.add('p2')
    assert patient_choices(session, actions, 'evacuate') == []


def test_treatment_prioritization_respects_visibility_proximity_and_offered_actions():
    session, _ = evacuation_session()
    session.current_state.characters[1].nearby = False
    direct = action('treat', 'TREAT_PATIENT', parameters={'treatment': 'Pressure bandage'})
    move = action('move', 'MOVE_TO')
    assert [r['patient_id'] for r in patient_choices(session, [direct], 'treat')] == ['p1']
    rows = patient_choices(session, [direct, move], 'treat')
    assert [r['patient_id'] for r in rows] == ['p1', 'p2']
    assert rows[1]['offered_actions'][0]['action_type'] == 'MOVE_TO'
    session.current_state.characters[1].unseen = True
    assert [r['patient_id'] for r in patient_choices(session, [direct, move], 'treat')] == ['p1']
    assert patient_choices(session, [action('end', 'END_SCENE')], 'treat') == []
    assert patient_choices(session, [direct], 'evacuate') == []


def test_cache_refreshes_on_evidence_target_and_eligibility_changes():
    session, _ = evacuation_session()
    provider = toolkit()
    provider.build_tools(session)
    first = provider.patient_assessment_text(session, 'evacuate')
    session.assessments.clear()
    assert provider.patient_assessment_text(session, 'evacuate') == first
    assert 'kaleido_patients' in session.assessments
    assert len(provider.kaleido.calls) == 1
    session.current_state.characters[0].tag = 'DELAYED'
    provider.patient_assessment_text(session, 'evacuate')
    assert len(provider.kaleido.calls) == 2
    session.alignment_target.kdma_values[0]['value'] = 0.0
    provider.patient_assessment_text(session, 'evacuate')
    assert len(provider.kaleido.calls) == 3
    session.evac_patients.add('p1')
    provider.patient_assessment_text(session, 'evacuate')
    assert len(provider.kaleido.calls[-1]) == 1
    provider.build_tools(session)
    provider.patient_assessment_text(session, 'evacuate')
    assert len(provider.kaleido.calls) == 5


def test_no_target_unsupported_attribute_and_no_candidates_do_not_invent_scores():
    session, _ = evacuation_session()
    provider = toolkit()
    session.alignment_target = None
    assert 'No scalar alignment target' in provider.patient_assessment_text(session, 'evacuate')
    session.alignment_target = SimpleNamespace(kdma_values=[{'kdma': 'unknown', 'value': .5}])
    output = provider.patient_assessment_text(session, 'evacuate')
    assert 'Unscored attributes' in output
    assert not provider.kaleido.calls
    assert all(not r['scores'] for r in session.assessments['kaleido_patients']['rows'])
    assert 'No eligible patient' in provider.patient_assessment_text(session, 'treat')


def test_legacy_action_mode_and_disabled_patient_explanations():
    session, _ = evacuation_session()
    provider = toolkit(assessment_subject='actions', patient_explain=False)
    provider.build_tools(session)
    assert provider.decision_context(session) is None  # one action template
    assert 'Closest to target' in provider.assessment_text(session)
    assert 'Model explanation' not in provider.patient_assessment_text(session, 'evacuate')
    assert provider.kaleido.explanations == 0
    with pytest.raises(ValueError, match='assessment_subject'):
        toolkit(assessment_subject='invalid')


def test_patient_mode_does_not_rank_generic_actions_when_no_patient_choice_exists():
    session, _ = session_for([action('end', 'END_SCENE'), action('search', 'SEARCH')])
    provider = toolkit(auto_assess=True)
    assert provider.decision_context(session) is None
    assert not provider.kaleido.calls


def test_partial_model_batch_is_not_published_as_a_complete_assessment():
    session, _ = evacuation_session()
    provider = toolkit()
    provider.kaleido.get_valence = lambda *args, **kwargs: [[.7, .1, .2]]
    with pytest.raises(ValueError, match='incomplete assessment batch'):
        provider.patient_assessment_text(session, 'evacuate')
    assert not session.assessments


def test_driver_binds_patient_tool_and_records_evidence_with_action():
    session, records = evacuation_session()
    provider = toolkit(auto_assess=False)
    driver = ITMOpenWorldLangChainDriver(extra_tools={'kaleido': provider})
    tools = {t.name: t for t in driver._build_tools(session)}
    tools['assess_patients'].invoke({'decision': 'evacuate'})
    assert 'assess_patients' in session.extra_tool_names
    chosen = action('evac', 'MOVE_TO_EVAC', character_id='p1')
    session.execute(chosen, justification='Use patient evidence')
    assert 'kaleido_patients' in json.dumps(records)
    assert len(session.scenario.executed) == 1


def test_hydra_kaleido_defaults_enable_patient_assessment():
    root = Path(__file__).resolve().parents[1] / 'configs'
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/phase2_langchain_agent_ow_kaleido_live'])
    assert cfg.driver.extra_tools.kaleido.assessment_subject == 'patients'
    assert not cfg.driver.extra_tools.kaleido.auto_assess
    assert cfg.driver.extra_tools.kaleido.patient_explain
    assert cfg.driver.extra_tools.kaleido.patient_prompt_format == 'full'


def test_hydra_compact_patient_prompt_override():
    root = Path(__file__).resolve().parents[1] / 'configs'
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/phase2_langchain_agent_ow_kaleido_live',
            'driver.extra_tools.kaleido.patient_prompt_format=compact'])
    assert cfg.driver.extra_tools.kaleido.patient_prompt_format == 'compact'
    assert not cfg.driver.extra_tools.kaleido.auto_assess


@pytest.mark.parametrize('decision', ['treat', 'evacuate'])
def test_compact_prompt_preserves_candidates_facts_and_value_definition(decision):
    session, _ = evacuation_session()
    session.current_state.characters[0].unstructured = (
        'Civilian likely responsible for the explosion; intent unknown.')
    session.current_state.characters[1].nearby = False
    session.current_state.characters[1].vitals = None
    session.treated_patients.add('p1')
    actions = [action('evac', 'MOVE_TO_EVAC'), action('move', 'MOVE_TO'),
               action('check', 'CHECK_VITALS')]
    rows = patient_choices(session, actions, decision)
    before = deepcopy(rows)
    compact = patient_query(session.current_state, rows, 0, decision, 'compact')
    full = patient_query(session.current_state, rows, 0, decision)
    assert len(compact) < len(full)
    assert 'Selected patient:\np1' in compact
    assert 'Other eligible patients:\np2' in compact
    assert 'Civilian likely responsible for the explosion; intent unknown.' in compact
    assert 'Proximity: distant' in compact and 'Vitals: unknown' in compact
    assert 'treated: recorded' in compact
    assert 'treated: not recorded' in compact
    assert 'Tag: IMMEDIATE' in compact and 'Laceration' in compact
    assert 'HIDDEN_GROUND_TRUTH' not in compact
    assert 'offered_actions' not in compact and 'first_seen' not in compact
    assert rows == before

    provider = toolkit(patient_prompt_format='compact')
    captured = []
    provider.kaleido.get_valence = lambda texts, vrds, descriptions, **kwargs: (
        captured.append((texts, vrds, descriptions)) or [[.7, .1, .2] for _ in texts])
    provider.patient_assessment_text(session, decision, actions)
    assert captured[0][0][0] == compact
    assert captured[0][1] == ['Value', 'Value']
    assert captured[0][2] == ['Favoring your own group'] * 2
    assert session.assessments['kaleido_patients']['patient_prompt_format'] == 'compact'


def test_compact_history_deduplicates_and_preserves_missing_facts_as_history():
    session, _ = evacuation_session()
    p1 = session.current_state.characters[0]
    p1.unstructured = 'Own-unit member. Leg wound.'
    session.patient_history.observe(session.current_state, action_count=1)
    p1.unstructured = 'Own-unit member. Leg wound. Bandage applied.'
    session.patient_history.observe(session.current_state, action_count=2)
    p1.unstructured = 'Leg wound. Bandage applied.'
    rows = patient_choices(session, session.refresh_actions(), 'evacuate')
    text = patient_query(session.current_state, rows, 0, 'evacuate', 'compact')
    assert text.count('Leg wound.') == 1
    assert text.count('Bandage applied.') == 1
    assert text.count('Own-unit member.') == 1
    assert 'Historical (may be stale): Own-unit member.' in text
    assert 'Current observations take precedence' in text


def test_compact_unseen_candidate_does_not_expose_hidden_current_facts():
    session, _ = evacuation_session()
    session.current_state.characters[0].unseen = True
    session.current_state.characters[0].unstructured = 'SECRET'
    session.current_state.characters[0].vitals = {'mental_status': 'SECRET'}
    provider = toolkit(patient_prompt_format='compact')
    provider.patient_assessment_text(session, 'evacuate')
    text = provider.kaleido.calls[0][0]
    assert 'SECRET' not in text and 'HIDDEN_GROUND_TRUTH' not in text
    assert 'Currently unseen; current clinical details unknown.' in text
    assert 'Historical (may be stale): Soldier in your unit, leg wound.' in text


def test_prompt_format_validation_and_cache_separation():
    with pytest.raises(ValueError, match='patient_prompt_format'):
        toolkit(patient_prompt_format='invalid')
    session, _ = evacuation_session()
    provider = toolkit()
    provider.patient_assessment_text(session, 'evacuate')
    provider.patient_prompt_format = 'compact'
    provider.patient_assessment_text(session, 'evacuate')
    provider.patient_assessment_text(session, 'evacuate')
    assert len(provider.kaleido.calls) == 2
    assert provider.kaleido.calls[0][0].startswith('Choosing patient')
    assert provider.kaleido.calls[1][0].startswith('Action: Prioritize patient')


def test_patient_prompt_compatibility_import_uses_toolkit_definition():
    from align_system.agent_tools.kaleido import KaleidoPatientHintPrompt
    from align_system.prompt_engineering.ow_prompts import (
        KaleidoPatientHintPrompt as LegacyPatientPrompt)

    assert LegacyPatientPrompt is KaleidoPatientHintPrompt
    assert 'assess_patients' in LegacyPatientPrompt()()


def test_patient_and_action_tools_keep_separate_cached_evidence_and_action_events():
    session, records = evacuation_session()
    events = []
    session.record_event = lambda event, **fields: events.append((event, fields))
    provider = toolkit()
    tools = {tool.name: tool for tool in provider.build_tools(session)}

    tools['assess_patients'].invoke({'decision': 'evacuate'})
    tools['assess_available_actions'].invoke({'justification': 'Compare actions'})
    expected = deepcopy(session.assessments)
    calls = len(provider.kaleido.calls)
    session.assessments.clear()

    tools['assess_patients'].invoke({'decision': 'evacuate'})
    tools['assess_available_actions'].invoke({'justification': 'Review actions'})
    assert session.assessments == expected
    assert len(provider.kaleido.calls) == calls
    assert [fields['cache_hit'] for name, fields in events
            if name == 'kaleido_assessment'] == [False, True]

    session.execute(action('evac', 'MOVE_TO_EVAC', character_id='p1'))
    assert records[0]['choice_info']['assessments'] == expected
