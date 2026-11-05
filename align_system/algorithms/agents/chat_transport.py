"""Chat transport capabilities and normalization shared by agent roles.

LangChain's provider-specific request hooks are kept here so dependency upgrades
can be checked without changing the review contract or workflow.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableBinding


def __getattr__(name):
    if name == 'CompatibleChatOpenAI':
        # Keep the OpenAI integration optional for text/Ollama-only callers.
        from .openai_chat_model import CompatibleChatOpenAI
        return CompatibleChatOpenAI
    raise AttributeError(name)


def chat_model_kind(model):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        pass
    else:
        if isinstance(model, ChatOpenAI):
            return 'openai'
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        pass
    else:
        if isinstance(model, ChatOllama):
            return 'ollama'
    return 'generic'


def effective_request_payload(model, messages, kwargs=None, *, kind=None):
    """Serialize the effective request, including final extra-body overrides."""
    bindings = []
    while isinstance(model, RunnableBinding):
        bindings.append(model.kwargs)
        model = model.bound
    merged = {}
    for binding in reversed(bindings):
        merged.update(binding)
    kwargs = deepcopy({**merged, **dict(kwargs or {})})
    kind = kind or chat_model_kind(model)
    if kind == 'openai':
        payload = deepcopy(model._get_request_payload(messages, **kwargs))
        extra = payload.pop('extra_body', None) or {}
        if not isinstance(extra, Mapping):
            raise ValueError('extra_body must be a mapping')
        payload.update(extra)
        return payload
    if kind == 'ollama':
        return deepcopy(model._chat_params(messages, **kwargs))
    defaults = {}
    for name in ('max_tokens', 'max_completion_tokens', 'max_output_tokens'):
        value = getattr(model, name, None)
        if value is not None:
            defaults[name] = value
    return {'messages': messages, **defaults, **kwargs}


@dataclass(frozen=True)
class ChatCapabilities:
    provider: str
    api: str

    def __post_init__(self):
        if self.provider not in ('vllm', 'openai'):
            raise ValueError('provider must be vllm or openai')
        if self.api not in ('chat_completions', 'responses'):
            raise ValueError('api must be chat_completions or responses')
        if self.provider == 'vllm' and self.api != 'chat_completions':
            raise ValueError('vllm requires Chat Completions routing')


def structured_review_kwargs(model, schema, *, provider, api=None):
    """Build constraints for an explicitly selected backend without mutation."""
    if chat_model_kind(model) != 'openai':
        raise ValueError('JSON schema review requires an unbound ChatOpenAI')
    params = model._default_params
    effective_api = ('responses' if model._use_responses_api(params)
                     else 'chat_completions')
    capabilities = ChatCapabilities(provider, api or effective_api)
    if capabilities.api != effective_api:
        raise ValueError('configured API does not match the chat model routing')
    body = params.get('extra_body')
    if body is None:
        body = {}
    if not isinstance(body, Mapping):
        raise ValueError('extra_body must be a mapping')
    for settings in (params, body):
        text = settings.get('text')
        if (any(key in ('response_format', 'structured_outputs', 'tools',
                        'tool_choice', 'functions', 'function_call')
                or key.startswith('guided_') for key in settings)
                or (isinstance(text, Mapping) and 'format' in text)):
            raise ValueError('review cannot combine preconfigured tools or output constraints')
    if capabilities.provider == 'vllm':
        return {'extra_body': {**deepcopy(dict(body)),
                               'structured_outputs': {'json': schema}}}
    return {'response_format': {
        'type': 'json_schema', 'json_schema': {
            'name': 'grounded_review', 'strict': True, 'schema': schema}}}


def normalize_chat_response(response):
    """Normalize reasoning extensions separately from visible answer content."""
    if not isinstance(response, AIMessage):
        return response
    extras = response.additional_kwargs
    reasoning = extras.get('reasoning_content')
    if not isinstance(reasoning, str):
        reasoning = extras.get('reasoning')
    if not isinstance(reasoning, str) or extras.get('reasoning_content') == reasoning:
        return response
    return response.model_copy(update={
        'additional_kwargs': {**deepcopy(extras), 'reasoning_content': reasoning}})


def response_text(response):
    """Extract public text only; reasoning and refusals never become advice."""
    if not isinstance(response, AIMessage):
        return None
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            block if isinstance(block, str) else block['text']
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get('type') in ('text', 'output_text')
                and isinstance(block.get('text'), str)))
    return ''


def response_failure(response):
    """Classify unusable completions before parsing even valid-looking JSON."""
    if not isinstance(response, AIMessage):
        return None
    blocks = response.content if isinstance(response.content, list) else []
    blocks = [block.get('value') if isinstance(block, Mapping)
              and block.get('type') == 'non_standard' else block for block in blocks]
    metadata = response.response_metadata
    if (response.additional_kwargs.get('refusal') is not None
            or metadata.get('finish_reason') == 'content_filter'
            or any(isinstance(block, Mapping) and block.get('type') == 'refusal'
                   for block in blocks)):
        return 'provider_refusal'
    if (metadata.get('status') == 'incomplete'
            or metadata.get('finish_reason') == 'length'):
        return 'response_shape'
    return None


def invoke_chat(model, messages, **kwargs):
    """Retain usable metadata when the SDK rejects an incomplete completion."""
    from openai import ContentFilterFinishReasonError, LengthFinishReasonError

    try:
        response = model.invoke(messages, **kwargs)
    except LengthFinishReasonError as exc:
        response = model._create_chat_result(exc.completion).generations[0].message
        response.response_metadata['finish_reason'] = 'length'
        if exc.completion.usage is not None:
            response.response_metadata['token_usage'] = exc.completion.usage.model_dump()
    except ContentFilterFinishReasonError:
        response = AIMessage(content='', additional_kwargs={'refusal': True},
                             response_metadata={'finish_reason': 'content_filter'})
    return normalize_chat_response(response)


def generation_metadata(response):
    """Allowlisted provider facts, without guessing why generation stopped."""
    if not isinstance(response, AIMessage):
        return {}
    metadata = response.response_metadata
    metadata = metadata if isinstance(metadata, Mapping) else {}
    generation = {}
    for name in ('finish_reason', 'status'):
        value = metadata.get(name)
        if isinstance(value, str) and value:
            generation[name] = value[:64]
    incomplete = metadata.get('incomplete_details')
    if isinstance(incomplete, Mapping) and isinstance(incomplete.get('reason'), str):
        generation['incomplete_reason'] = incomplete['reason'][:64]
    normalized = response.usage_metadata
    normalized = normalized if isinstance(normalized, Mapping) else {}
    provider = metadata.get('token_usage')
    provider = provider if isinstance(provider, Mapping) else {}

    def count(name, *values):
        for value in values:
            if type(value) is int and 0 <= value < 2 ** 63:
                generation[name] = value
                return

    for name, provider_name in (
            ('input_tokens', 'prompt_tokens'),
            ('output_tokens', 'completion_tokens'),
            ('total_tokens', 'total_tokens')):
        count(name, normalized.get(name), provider.get(provider_name))
    details = normalized.get('output_token_details')
    details = details if isinstance(details, Mapping) else {}
    provider_details = provider.get('completion_tokens_details')
    provider_details = provider_details if isinstance(provider_details, Mapping) else {}
    count('reasoning_tokens', details.get('reasoning'),
          provider_details.get('reasoning_tokens'))
    input_details = normalized.get('input_token_details')
    input_details = input_details if isinstance(input_details, Mapping) else {}
    provider_input = provider.get('prompt_tokens_details')
    provider_input = provider_input if isinstance(provider_input, Mapping) else {}
    count('cached_input_tokens', input_details.get('cache_read'),
          provider_input.get('cached_tokens'))
    return generation
