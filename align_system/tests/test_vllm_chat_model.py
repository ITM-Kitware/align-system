"""Offline checks of the real LangChain/HTTP request path and context guard."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path

import httpx
from hydra.utils import instantiate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from omegaconf import OmegaConf
import pytest

from align_system.exceptions import ContextWindowError
from align_system.utils.vllm_chat_model import VLLMChatOpenAI


TOOLS = [{'type': 'function', 'function': {
    'name': 'observe', 'description': 'Observe the environment',
    'parameters': {'type': 'object', 'properties': {}}}}]


def exchange(label, results=1):
    return [AIMessage(content='', tool_calls=[
        {'name': 'observe', 'args': {}, 'id': f'{label}-{i}'}
        for i in range(results)]), *[
        ToolMessage(content=label, tool_call_id=f'{label}-{i}')
        for i in range(results)]]


class Server:
    """Mock token counts, but use the real SDK serialization and transport."""

    def __init__(self, count=None, limit=16384, status=200):
        self.count = count or (lambda body: 100)
        self.limit = limit
        self.status = status
        self.tokenizations = []
        self.completions = []

    def handle(self, request):
        body = json.loads(request.content)
        assert request.headers['authorization'] == 'Bearer EMPTY'
        if request.url.path == '/proxy/tokenize':
            self.tokenizations.append(body)
            return httpx.Response(self.status, json={
                'count': self.count(body), 'max_model_len': self.limit,
                'tokens': [], 'error': {'message': 'Unavailable'}})
        assert request.url.path == '/proxy/v1/chat/completions'
        self.completions.append(body)
        content = '{"score": 80}' if 'response_format' in body else 'Ready'
        return httpx.Response(200, json={
            'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': 'Qwen/test', 'choices': [{
                'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content}}]})


@pytest.fixture
def make_model():
    clients = []

    def make(server, **kwargs):
        transport = httpx.MockTransport(server.handle)
        sync_client = httpx.Client(transport=transport)
        async_client = httpx.AsyncClient(transport=transport)
        clients.extend([sync_client, async_client])
        return VLLMChatOpenAI(
            model='Qwen/test', api_key='EMPTY',
            base_url='http://test/proxy/v1/',
            http_client=sync_client, http_async_client=async_client,
            max_tokens=2048, **kwargs)

    yield make
    for client in clients:
        if isinstance(client, httpx.AsyncClient):
            asyncio.run(client.aclose())
        else:
            client.close()


def test_under_budget_counts_exact_tools_and_template_options(make_model):
    server = Server()
    model = make_model(server, extra_body={
        'chat_template_kwargs': {'enable_thinking': False}})
    messages = [SystemMessage('Instructions'), HumanMessage('Begin')]
    result = model.bind_tools(TOOLS).invoke(messages)
    assert result.content == 'Ready'
    assert len(server.tokenizations) == len(server.completions) == 1
    counted, generated = server.tokenizations[0], server.completions[0]
    for field in ('messages', 'tools', 'model', 'chat_template_kwargs'):
        assert counted[field] == generated[field]
    assert generated['max_completion_tokens'] == 2048


def test_trims_complete_old_exchanges_keeps_latest_and_original(make_model, caplog):
    caplog.set_level('INFO', logger='align_system.utils.vllm_chat_model')
    server = Server(count=lambda body: 15000 if any(
        m.get('content') == 'old' for m in body['messages']) else 10000)
    model = make_model(server)
    opening = [SystemMessage('Instructions'), HumanMessage('Begin')]
    latest = exchange('latest', results=2)
    messages = opening + exchange('old', results=2) + latest
    original = deepcopy(messages)
    model.bind_tools(TOOLS).invoke(messages)
    assert messages == original
    assert len(server.tokenizations) == 2
    assert server.completions[0]['messages'] == model._get_request_payload(
        opening + latest)['messages']
    assert 'trimmed 3 old messages' in caplog.text


@pytest.mark.parametrize('count,should_fit', [(14080, True), (14081, False)])
def test_reserves_response_and_safety_margin(make_model, count, should_fit):
    server = Server(count=lambda _: count)
    model = make_model(server)
    if should_fit:
        model.invoke('Current observation')
        assert len(server.completions) == 1
    else:
        with pytest.raises(ContextWindowError, match='server limit 16384'):
            model.invoke('Current observation')
        assert not server.completions
    assert len(server.tokenizations) == 1


def test_oversized_latest_exchange_never_discarded(make_model):
    server = Server(count=lambda _: 15000)
    model = make_model(server)
    messages = [SystemMessage('Instructions'), HumanMessage('Begin'),
                *exchange('old'), *exchange('large latest observation')]
    with pytest.raises(ContextWindowError, match='No completion was sent'):
        model.invoke(messages)
    assert len(server.tokenizations) == 2
    assert server.tokenizations[-1]['messages'][-1]['content'] == 'large latest observation'
    assert not server.completions


def test_output_override_uses_server_limit(make_model):
    server = Server(count=lambda _: 6000, limit=8192)
    model = make_model(server)
    model.invoke('Observation', max_tokens=1024)
    assert server.completions[0]['max_completion_tokens'] == 1024
    with pytest.raises(ContextWindowError, match='server limit 8192'):
        model.invoke('Observation')
    assert len(server.completions) == 1


@pytest.mark.parametrize('status', [400, 404, 422])
def test_tokenizer_error_fails_before_completion(make_model, status):
    server = Server(status=status)
    with pytest.raises(ContextWindowError, match=f'HTTP {status}'):
        make_model(server).invoke('Observation')
    assert len(server.tokenizations) == 1
    assert not server.completions


def test_invalid_tokenizer_response_fails_closed(make_model):
    server = Server(limit=None)
    with pytest.raises(ContextWindowError, match='valid count and max_model_len'):
        make_model(server).invoke('Observation')
    assert not server.completions


def test_structured_batch_is_budgeted(make_model):
    server = Server()
    model = make_model(server)
    schema = {'title': 'Scores', 'type': 'object',
              'properties': {'score': {'type': 'integer'}}, 'required': ['score']}
    scorer = model.with_structured_output(schema, method='json_schema')
    result = scorer.batch([[SystemMessage('Score actions'), HumanMessage('Choices')]] * 2)
    assert result == [{'score': 80}] * 2
    assert len(server.tokenizations) == len(server.completions) == 2
    assert all(body['max_completion_tokens'] == 2048 for body in server.completions)


class CRServer(Server):
    """Reproduce a truncated CR completion through the real SDK parser."""

    def handle(self, request):
        response = super().handle(request)
        if request.url.path != '/proxy/v1/chat/completions':
            return response
        body = json.loads(request.content)
        if 'response_format' not in body:
            return response
        completion = response.json()
        choice = completion['choices'][0]
        if body['max_completion_tokens'] <= 2048:
            choice['finish_reason'] = 'length'
            choice['message']['content'] = '{"reasoning": "Incomplete'
            completion['usage'] = {
                'completion_tokens': 2048, 'prompt_tokens': 884, 'total_tokens': 2932}
        else:
            schema = body['response_format']['json_schema']['schema']
            choice['message']['content'] = json.dumps({
                key: {'score': 80,
                      'explanation': 'The response favors the same group.'}
                for key in schema['properties']})
        return httpx.Response(200, json=completion)


@pytest.mark.parametrize('budget,failed', [(None, True), (8192, False)])
def test_cr_output_budget_handles_length_error_without_changing_actor(
        make_model, caplog, budget, failed):
    from align_system.tests.test_agent_assessments import session_for, toolkit_for

    server = CRServer()
    model = make_model(server)
    toolkit = toolkit_for('comparative_regression', 'evidence',
                          generation_kwargs={} if budget is None else {'max_tokens': budget})
    toolkit.chat_model = model
    session, _ = session_for()
    toolkit.assessment_text(session)
    record = session.assessments['comparative_regression']
    assert bool(record['failed']) == failed
    if failed:
        assert 'length limit was reached' in caplog.text
    else:
        assert record['rows'][0]['scores']['affiliation']['predicted'] == 0.8
        assert record['reasonings']['affiliation']
    assert server.completions[0]['max_completion_tokens'] == (budget or 2048)
    model.invoke('Continue the agent turn')
    assert server.completions[-1]['max_completion_tokens'] == 2048


def test_cr_output_override_reserves_context_before_request(make_model, caplog):
    from align_system.tests.test_agent_assessments import session_for, toolkit_for

    server = CRServer(count=lambda _: 884, limit=9000)
    model = make_model(server)
    toolkit = toolkit_for('comparative_regression', 'evidence',
                          generation_kwargs={'max_tokens': 8192})
    toolkit.chat_model = model
    session, _ = session_for()
    toolkit.assessment_text(session)
    assert session.assessments['comparative_regression']['failed'] == ['affiliation']
    assert '8192 output tokens' in caplog.text
    assert not server.completions
    model.invoke('Continue the agent turn')
    assert len(server.completions) == 1


@pytest.mark.parametrize('config_factory', [dict, OmegaConf.create], ids=['dict', 'hydra'])
def test_cr_disables_thinking_for_scorer_without_changing_actor(make_model, config_factory):
    from align_system.tests.test_agent_assessments import session_for, toolkit_for

    server = CRServer()
    model = make_model(server, extra_body={
        'chat_template_kwargs': {'enable_thinking': True}})
    generation_kwargs = config_factory({
        'max_tokens': 8192,
        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}})
    toolkit = toolkit_for('comparative_regression', generation_kwargs=generation_kwargs)
    toolkit.chat_model = model
    session, _ = session_for()
    toolkit.assessment_text(session)
    assert not session.assessments['comparative_regression']['failed']
    assert len(server.tokenizations) == len(server.completions) == 1
    assert server.completions[-1]['chat_template_kwargs']['enable_thinking'] is False
    assert server.tokenizations[-1]['chat_template_kwargs']['enable_thinking'] is False
    model.invoke('Continue the agent turn')
    assert server.completions[-1]['chat_template_kwargs']['enable_thinking'] is True
    assert server.tokenizations[-1]['chat_template_kwargs']['enable_thinking'] is True
    assert model.extra_body['chat_template_kwargs']['enable_thinking'] is True
    assert generation_kwargs['extra_body']['chat_template_kwargs']['enable_thinking'] is False


def test_cr_context_guard_preserves_all_selected_examples(make_model):
    from align_system.tests.test_agent_assessments import session_for, toolkit_for
    from align_system.tests.test_agent_compact_regression import TwentyExamples

    examples = TwentyExamples()
    server = CRServer(count=lambda body: 15000)
    toolkit = toolkit_for('comparative_regression', icl=examples)
    toolkit.chat_model = make_model(server)
    session, _ = session_for()
    text = toolkit.assessment_text(session)
    record = session.assessments['comparative_regression']
    assert record['failed'] == ['affiliation']
    assert len(record['scoring_attempts']['affiliation']) == 1
    assert len(server.tokenizations) == 1
    assert not server.completions
    assert [message['content'] for message in server.tokenizations[0]['messages'][1:-1]] == [
        example.content for example in examples.dialog]
    for example in examples.dialog:
        assert example.content in text


def test_async_and_stream_paths_are_budgeted(make_model):
    server = Server()
    model = make_model(server)
    assert asyncio.run(model.ainvoke('Observation')).content == 'Ready'
    assert list(model.stream('Observation'))[0].content == 'Ready'
    assert len(server.tokenizations) == len(server.completions) == 2


def test_qwen_config_enables_guard_and_response_limit():
    path = Path(__file__).parents[1] / 'configs/driver/chat_model/vllm_qwen38_27b.yaml'
    model = instantiate(OmegaConf.load(path))
    assert isinstance(model, VLLMChatOpenAI)
    assert model.max_tokens == 8192


def test_pr_request_budget_preserves_evidence_through_adapter(make_model):
    from align_system.drivers.agent_context_budget import RequestBudget

    server = Server()
    model = make_model(server)
    bound = model.bind_tools(TOOLS)
    messages = [SystemMessage('Instructions'), HumanMessage('Current evidence')]
    request = RequestBudget(bound).prepare(messages)
    assert request.invoke_kwargs['preserve_messages'] is True
    bound.invoke(request.messages, **request.invoke_kwargs)
    assert len(server.completions) == 1
    assert server.tokenizations[0]['messages'] == server.completions[0]['messages']
    assert 'preserve_messages' not in server.completions[0]
