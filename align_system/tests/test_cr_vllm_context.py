"""CR must retain every selected training message or fail before generation."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

from align_system.exceptions import ContextWindowError
from align_system.tests.test_vllm_chat_model import Server, make_model


def example_messages():
    return [SystemMessage('Score the current actions.'), *[
        message for index in range(20) for message in (
            HumanMessage(f'Training scenario {index + 1}'),
            AIMessage(f'Labeled score {index + 1}'))],
        HumanMessage('Current scene and actions')]


@pytest.mark.parametrize('asynchronous', [False, True])
def test_preserve_messages_rejects_overflow_without_dropping_examples(make_model, asynchronous):
    server = Server(count=lambda body: 15000 if len(body['messages']) > 3 else 100)
    model = make_model(server)
    messages = example_messages()
    with pytest.raises(ContextWindowError, match='No messages were trimmed'):
        if asynchronous:
            asyncio.run(model.ainvoke(messages, preserve_messages=True))
        else:
            model.invoke(messages, preserve_messages=True)
    assert len(server.tokenizations) == 1
    assert len(server.tokenizations[0]['messages']) == 42
    assert not server.completions

    # A later actor request retains its existing history-trimming behavior.
    model.invoke(messages)
    assert len(server.completions) == 1
    assert len(server.completions[0]['messages']) == 3


@pytest.mark.parametrize('asynchronous', [False, True])
def test_preserve_messages_keeps_all_examples_and_never_reaches_api(make_model, asynchronous):
    server = Server()
    model = make_model(server)
    messages = example_messages()
    if asynchronous:
        asyncio.run(model.ainvoke(messages, preserve_messages=True))
    else:
        model.invoke(messages, preserve_messages=True)
    assert len(server.completions) == 1
    assert len(server.completions[0]['messages']) == 42
    assert 'preserve_messages' not in server.completions[0]
    assert 'preserve_messages' not in server.tokenizations[0]
