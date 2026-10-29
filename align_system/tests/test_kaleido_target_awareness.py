"""A target sweep must refresh rankings, actor context, and recorded evidence."""
from copy import deepcopy
import json

from langchain_core.messages import AIMessage
import pytest

from align_system.agent_tools.kaleido import KaleidoToolkit
from align_system.algorithms.target_context import TargetContext
from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver
from align_system.tests.test_agent_assessments import action, action_call, session_for


class FixedPredictions:
    """Keep estimates fixed so only the requested target changes the ranking."""

    def __init__(self):
        self.calls = []

    def get_relevance(self, texts, vrds, descriptions, **kwargs):
        self.calls.append({'texts': list(texts), 'descriptions': list(descriptions)})
        return [[0.9, 0.1] for _ in texts]

    def get_valence(self, texts, *args, **kwargs):
        return [[0.05, 0.85, 0.10] if text.startswith('Choosing to "Low option"')
                else [0.85, 0.05, 0.10] for text in texts]


def target_session(attribute='merit', value=0.0):
    session, records = session_for([
        action('low', unstructured='Low option'),
        action('high', unstructured='High option')])
    session.alignment_target.id = f'{attribute}-{value}'
    session.alignment_target.kdma_values = [{'kdma': attribute, 'value': value}]
    events = []
    session.record_event = lambda event, **fields: events.append(
        deepcopy({'event': event, **fields}))
    return session, records, events


def toolkit():
    return KaleidoToolkit(FixedPredictions(), {
        'merit': {'description': 'Favor victims over perpetrators'},
        'affiliation': {'description': 'Favor members of your group'},
    }, assessment_subject='actions')


@pytest.mark.parametrize('attribute', ['merit', 'affiliation'])
def test_changing_target_reverses_closest_action_with_fixed_predictions(attribute):
    provider = toolkit()
    session, _, _ = target_session(attribute)
    before = deepcopy(session.current_state.to_dict())
    choices = deepcopy([a.to_dict() for a in session.scenario.actions])

    low_text = provider.assessment_text(session)
    low = deepcopy(session.assessments['kaleido'])
    session.alignment_target.kdma_values[0]['value'] = 1.0
    high_text = provider.assessment_text(session)
    high = session.assessments['kaleido']

    assert 'Closest to target overall: search: Low option' in low_text
    assert 'Closest to target overall: search: High option' in high_text
    assert low['targets'] == {attribute: 0.0}
    assert high['targets'] == {attribute: 1.0}
    assert [row['scores'][attribute]['estimate'] for row in low['rows']] == pytest.approx([0.1, 0.9])
    assert [row['scores'][attribute]['estimate'] for row in high['rows']] == pytest.approx([0.1, 0.9])
    assert [row['scores'][attribute]['distance'] for row in low['rows']] == pytest.approx([0.1, 0.9])
    assert [row['scores'][attribute]['distance'] for row in high['rows']] == pytest.approx([0.9, 0.1])
    assert provider.kaleido.calls[0] == provider.kaleido.calls[1]
    assert session.current_state.to_dict() == before
    assert [a.to_dict() for a in session.scenario.actions] == choices


@pytest.mark.parametrize('change', ['state', 'action_id', 'description', 'parameters'])
def test_cached_assessment_does_not_survive_changed_scoring_inputs(change):
    provider = toolkit()
    session, _, _ = target_session()
    provider.assessment_text(session)
    provider.assessment_text(session)
    assert len(provider.kaleido.calls) == 1

    if change == 'state':
        session.current_state.unstructured = 'The situation has changed.'
    elif change == 'action_id':
        session.scenario.actions[0].action_id = 'replacement'
    elif change == 'description':
        session.scenario.actions[0].unstructured = 'A different option'
    else:
        session.scenario.actions[0].parameters = {'destination': 'new area'}

    assert session.n_actions == 0
    provider.assessment_text(session)
    assert len(provider.kaleido.calls) == 2


def test_cached_evidence_is_restored_and_detached_from_later_actions():
    provider = toolkit()
    session, _, events = target_session()
    text = provider.assessment_text(session)
    original = deepcopy(session.assessments['kaleido'])
    session.assessments.clear()
    assert provider.assessment_text(session) == text
    assert session.assessments['kaleido'] == original
    assert len(provider.kaleido.calls) == 1
    assessments = [e for e in events if e['event'] == 'kaleido_assessment']
    assert [e['cache_hit'] for e in assessments] == [False, True]
    assert all(e['targets'] == {'merit': 0.0} for e in assessments)
    assert all(e['alignment_target_id'] == 'merit-0.0' for e in assessments)
    session.scenario.actions[0].parameters['destination'] = 'later change'
    assert session.assessments['kaleido'] == original
    json.dumps(assessments)


def test_two_episode_execution_records_and_events_use_each_current_target():
    provider = toolkit()
    actor_contexts = []

    class Actor:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            actor_contexts.append(deepcopy(messages))
            current_target, _ = json.JSONDecoder().raw_decode(
                messages[0].content.split('TARGET_JSON=', 1)[1])
            action_id = 'low' if current_target['merit'] == 0.0 else 'high'
            return AIMessage(content='', tool_calls=[action_call(action_id)])

    class TargetBlock:
        def render(self, target):
            return 'TARGET_JSON=' + json.dumps({e['kdma']: e['value'] for e in target.kdma_values})

    driver = ITMOpenWorldLangChainDriver(
        chat_model=Actor(), extra_tools={'kaleido': provider},
        target_context=TargetBlock(), apply_action_filtering=False,
        max_actions_per_scenario=1)
    for value, selected in [(0.0, 'low'), (1.0, 'high')]:
        session, records, events = target_session(value=value)
        driver._run_agent_loop(session)
        assert len(records) == 1
        record = records[0]
        assessment = record['choice_info']['assessments']['kaleido']
        assert record['output']['action']['action_id'] == selected
        assert record['choice_info']['alignment_target_values'] == assessment['targets'] == {'merit': value}
        assert assessment['alignment_target_id'] == record['input']['alignment_target_id']
        assert assessment['n_actions'] == 0
        assert assessment['mode'] == 'ranked'
        assert assessment['scoring_scope'] == 'offered_action_templates'
        assert assessment['auto_assess'] is True
        assert assessment['unscored'] == []
        assert any(assessment['text'] in str(m.content) for m in actor_contexts[-1])
        assessment_event = next(e for e in events if e['event'] == 'kaleido_assessment')
        assert assessment_event['targets'] == assessment['targets']
        assert assessment_event['alignment_target_id'] == assessment['alignment_target_id']
        assert assessment_event['decision_index'] == assessment['n_actions']
        assert assessment_event['assessment']['targets'] == assessment['targets']
        json.dumps(record)
    assert len(provider.kaleido.calls) == 2


def test_pinned_actor_target_refreshes_even_without_a_successful_action():
    session, records, events = target_session()
    provider = toolkit()
    contexts = []

    class Actor:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            contexts.append(deepcopy(messages))
            if len(contexts) == 1:
                session.alignment_target.kdma_values[0]['value'] = 1.0
                return AIMessage(content='Reconsidering the current target.')
            return AIMessage(content='', tool_calls=[action_call('high')])

    driver = ITMOpenWorldLangChainDriver(
        chat_model=Actor(), extra_tools={'kaleido': provider},
        target_context=TargetContext({
            'merit': {'description': 'Favor victims over perpetrators'}}),
        apply_action_filtering=False, max_messages_in_context=2,
        max_actions_per_scenario=1)
    driver._run_agent_loop(session)
    assert len(contexts) == 2
    assert '"value": 0.0' in contexts[0][0].content
    assert '"value": 1.0' in contexts[1][0].content
    assert '"value": 0.0' not in contexts[1][0].content
    assert all('Favor victims over perpetrators' in context[0].content for context in contexts)
    assert records[0]['choice_info']['alignment_target_values'] == {'merit': 1.0}
    assert records[0]['choice_info']['assessments']['kaleido']['targets'] == {'merit': 1.0}
    assert len(provider.kaleido.calls) == 2
    assessments = [e for e in events if e['event'] == 'kaleido_assessment']
    assert [e['targets'] for e in assessments] == [{'merit': 0.0}, {'merit': 1.0}]
    assert any('Closest to target overall: search: High option' in str(m.content)
               for m in contexts[1])
