"""Complete CR results and selected evidence must reach the optional actor tool."""

from copy import deepcopy
import json

import pytest

from align_system.agent_tools.comparative_regression import ComparativeRegressionToolkit
from align_system.data_models.attribute import Attribute
from align_system.data_models.dialog import DialogElement
from align_system.exceptions import ContextWindowError
from align_system.prompt_engineering.outlines_prompts import ComparativeRegressionSchema
from align_system.tests.test_agent_assessments import session_for, toolkit_for
from align_system.tests.test_agent_concrete_actions import treatment_session


def complete_response(score=80):
    return {key: {'score': score, 'explanation': f'Evidence for {key}.'}
            for key in ('C01', 'C02')}


class ScriptedScorer:
    """Return planned batches while retaining the actual request contract."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def with_structured_output(self, schema, method):
        assert method == 'json_schema'
        self.schema = deepcopy(schema)
        return self

    def batch(self, messages, **kwargs):
        self.requests.append({'schema': self.schema, 'messages': deepcopy(messages),
                              'kwargs': deepcopy(kwargs)})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


def use_scorer(responses, **kwargs):
    toolkit = toolkit_for('comparative_regression', **kwargs)
    toolkit.chat_model = ScriptedScorer(responses)
    return toolkit


def record_for(toolkit, session=None):
    session = session or session_for()[0]
    text = toolkit.assessment_text(session)
    return text, session.assessments['comparative_regression']


def add_target(toolkit, session, name='merit'):
    toolkit.attributes[name] = Attribute(
        name=f'{name} focus', kdma=name, factor=100,
        description=f'Assess {name}', score_examples='An example scores 80.')
    toolkit.score_schema_template = ComparativeRegressionSchema({
        'affiliation focus': 100, f'{name} focus': 100})
    session.alignment_target.kdma_values.append({'kdma': name, 'value': 0.8})


def test_compact_schema_keeps_concrete_patient_details_in_query():
    toolkit = use_scorer([[complete_response()]])
    text, record = record_for(toolkit, treatment_session())
    request = toolkit.chat_model.requests[0]
    schema = request['schema']
    assert list(schema['properties']) == ['C01', 'C02']
    assert set(schema['required']) == {'C01', 'C02'}
    assert schema['additionalProperties'] is False
    for candidate in schema['properties'].values():
        assert set(candidate['required']) == {'score', 'explanation'}
        assert candidate['properties']['score']['minimum'] == 0
        assert candidate['properties']['score']['maximum'] == 100
        assert candidate['properties']['explanation']['maxLength'] == 160
    query = request['messages'][0][-1].content
    assert all(token in query for token in ('C01', 'C02', 'p1', 'p2',
                                             'Tourniquet', 'left thigh', 'action_id'))
    assert 'Evidence for C01.' in text and 'Evidence for C02.' in text
    assert record['rows'][0]['scores']['affiliation']['explanations'] == ['Evidence for C01.']


@pytest.mark.parametrize('score', [True, False, '80', None, float('nan'),
                                  float('inf'), float('-inf'), -1, 101])
def test_invalid_score_is_retried_and_never_reaches_record(score):
    invalid = complete_response()
    invalid['C01']['score'] = score
    toolkit = use_scorer([[invalid], [complete_response(20)]])
    _, record = record_for(toolkit)
    assert len(toolkit.chat_model.requests) == 2
    assert not record['failed']
    assert all(row['scores']['affiliation']['predicted'] == 0.2
               for row in record['rows'])
    assert [a['status'] for a in record['scoring_attempts']['affiliation']] == [
        'failed', 'complete']
    assert record['scoring_attempts']['affiliation'][0]['error']
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize('invalid', [
    {'C01': {'score': 80, 'explanation': 'Only one choice.'}},
    {'C01': {'explanation': 'Missing score.'},
     'C02': {'score': 80, 'explanation': 'Present.'}},
    {'C01': {'score': 80, 'explanation': ''},
     'C02': {'score': 80, 'explanation': 'Present.'}},
    {'C01': {'score': 80, 'explanation': 10},
     'C02': {'score': 80, 'explanation': 'Present.'}},
    {'C01': {'score': 80, 'explanation': 'x' * 161},
     'C02': {'score': 80, 'explanation': 'Present.'}},
    None,
])
def test_incomplete_candidate_payload_is_retried(invalid):
    toolkit = use_scorer([[invalid], [complete_response()]])
    _, record = record_for(toolkit)
    assert len(toolkit.chat_model.requests) == 2
    assert record['failed'] == []
    assert all(row['scores']['affiliation']['predicted'] == 0.8
               for row in record['rows'])


@pytest.mark.parametrize('samples', [[], [complete_response()],
                                     [complete_response()] * 3])
def test_wrong_sample_count_is_retried_before_averaging(samples):
    toolkit = use_scorer([samples, [complete_response(0), complete_response(100)]],
                         num_samples=2)
    _, record = record_for(toolkit)
    assert len(toolkit.chat_model.requests) == 2
    assert all(len(request['messages']) == 2 for request in toolkit.chat_model.requests)
    assert all(row['scores']['affiliation']['predicted'] == 0.5
               for row in record['rows'])


def test_retries_only_failed_attribute_and_reports_exhausted_candidates():
    session, _ = session_for()
    toolkit = use_scorer([[complete_response()], [{}], [{}], [{}]])
    add_target(toolkit, session)
    text, record = record_for(toolkit, session)
    assert len(toolkit.chat_model.requests) == 4
    assert record['failed'] == ['merit']
    assert len(record['scoring_attempts']['affiliation']) == 1
    assert len(record['scoring_attempts']['merit']) == 3
    assert record['missing_scores'] == {'merit': ['C01', 'C02']}
    assert all('affiliation' in row['scores'] and 'merit' not in row['scores']
               for row in record['rows'])
    assert all(row['avg_distance'] is None for row in record['rows'])
    assert 'Closest to target overall:' not in text
    assert 'merit' in text and 'C01' in text and 'C02' in text


def test_missing_target_definition_prevents_partial_overall_recommendation():
    session, _ = session_for()
    session.alignment_target.kdma_values.append({'kdma': 'unknown', 'value': 0.2})
    toolkit = use_scorer([[complete_response()]])
    text, record = record_for(toolkit, session)
    assert record['unscored'] == ['unknown']
    assert all(row['avg_distance'] is None for row in record['rows'])
    assert 'Closest to target overall:' not in text


def test_context_failure_does_not_repeat_identical_oversized_request():
    toolkit = use_scorer([ContextWindowError('Newest observation exceeds context')])
    _, record = record_for(toolkit)
    assert len(toolkit.chat_model.requests) == 1
    assert record['failed'] == ['affiliation']
    assert record['missing_scores'] == {'affiliation': ['C01', 'C02']}


@pytest.mark.parametrize('max_retries, attempts', [(0, 1), (1, 2)])
def test_configured_retry_limit_is_respected(max_retries, attempts):
    toolkit = use_scorer([[{}]] * attempts, max_retries=max_retries)
    _, record = record_for(toolkit)
    assert len(toolkit.chat_model.requests) == attempts
    assert record['failed'] == ['affiliation']


class TwentyExamples:
    def __init__(self):
        self.calls = 0
        self.dialog = []
        self.selected = []
        for number in range(1, 21):
            prompt = (f'Scenario E{number:02}: Historical event {number}.\n'
                      f'Candidate A: Treat historical patient {number}.\n'
                      f'Candidate B: Check historical patient {number}.')
            response = f'Candidate A: affiliation={number}; Candidate B: affiliation=80.'
            self.dialog.extend([DialogElement(role='user', content=prompt),
                                DialogElement(role='assistant', content=response)])
            self.selected.append({'prompt': prompt, 'response': response,
                                  'similarity_score': 1 - number / 100})

    def run(self, state, choices, actions, alignment_target):
        self.calls += 1
        # Similarity selection must retain the original descriptive query.
        assert all('action_id' in choice for choice in choices)
        return {'affiliation': self.dialog}, {'affiliation': self.selected}


@pytest.mark.parametrize('scoring_failed', [False, True])
def test_all_twenty_selected_examples_reach_actor_even_when_scoring_fails(scoring_failed):
    examples = TwentyExamples()
    responses = [[{}]] * 3 if scoring_failed else [[complete_response()]]
    toolkit = use_scorer(responses, icl=examples)
    session, _ = session_for()
    tool = toolkit.build_tools(session)[0]
    text = tool.invoke({'justification': 'Review the selected evidence.'})
    record = session.assessments['comparative_regression']
    assert examples.calls == 1
    assert len(record['icl_examples']['affiliation']) == 20
    assert 'training' in text.lower() and 'current' in text.lower()
    assert '20' in text
    for request in toolkit.chat_model.requests:
        scorer_examples = request['messages'][0][1:-1]
        assert [message.content for message in scorer_examples] == [
            example.content for example in examples.dialog]
    for example in examples.dialog:
        assert example.content in text
    if not scoring_failed:
        assert tool.invoke({'justification': 'Re-read.'}) == text
        assert examples.calls == 1
        assert len(toolkit.chat_model.requests) == 1


@pytest.mark.parametrize('mode', ['evidence', 'ranked'])
def test_default_toolkit_leaves_cr_optional(mode):
    reference = toolkit_for('comparative_regression', mode=mode)
    toolkit = ComparativeRegressionToolkit(
        chat_model=reference.chat_model,
        attributes=reference.attributes,
        scenario_description_template=reference.scenario_description_template,
        prompt_template=reference.prompt_template,
        score_schema_template=reference.score_schema_template,
        assessment_mode=mode)
    session, _ = session_for()
    assert toolkit.auto_assess is False
    assert 'CR REMINDER' in toolkit.decision_context(session)
    assert toolkit.chat_model.calls == 0
    assert 'optional' in toolkit.prompt_hint.lower()
    assert 'before choosing each environment action, call' not in toolkit.prompt_hint.lower()
