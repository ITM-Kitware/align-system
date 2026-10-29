"""Per-request token capacity checks without mutating shared LangChain models.

Counting is deliberately separate from invoking: preflights are not model attempts.
Provider discovery is lazy, and errors never include prompt or credential contents.
Long scenarios accumulate messages, patient history, tool schemas, and review
evidence. Reserve output capacity before generation, and let the caller discard
only optional conversation history when the complete request would not fit.
This token capacity is independent of the workflow's model-call limit.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
import re
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict

from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableBinding


class ContextBudgetError(RuntimeError):
    def __init__(self, reason, diagnostics=None):
        self.diagnostics = {**(diagnostics or {}), 'reason': reason}
        super().__init__(f"Context budget: {reason}.")


@dataclass(frozen=True)
class BudgetMeasurement:
    input_tokens: int
    context_window: int
    output_tokens: int
    headroom: int
    input_limit: int
    method: str
    invoke_kwargs: dict
    diagnostics: dict

    @property
    def fits(self):
        return self.input_tokens <= self.input_limit


@dataclass(frozen=True)
class PreparedRequest:
    messages: list
    invoke_kwargs: dict
    diagnostics: dict


def _positive(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _unwrap(model):
    bindings = []
    while isinstance(model, RunnableBinding):
        bindings.append(model.kwargs)
        model = model.bound
    kwargs = {}
    for binding in reversed(bindings):
        kwargs.update(binding)
    return model, kwargs


class ContextBudgetManager:
    """Create scenario-owned role budgets; ``None`` preserves Python defaults.

    Callers retain the returned budget for one scenario. The factory never
    retains models or tool bindings from completed scenarios.
    """

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.enabled = bool(config is not None and self.config.get('enabled', True))

    def for_model(self, model, role):
        if not self.enabled:
            return None
        return RequestBudget(model, role, self.config.get(role) or {})


class RequestBudget:
    def __init__(self, model, role='decision', config=None):
        self.model, self.bound_kwargs = _unwrap(model)
        self.role = role
        self.config = dict(config or {})
        self.counter = self.config.get('counter', 'auto')
        self._tokenizer_supported = None
        self._serving_window = None

    def _error(self, reason, **diagnostics):
        return ContextBudgetError(reason, {'role': self.role, **diagnostics})

    def _kind(self):
        # Local imports keep unrelated provider integrations optional.
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            pass
        else:
            if isinstance(self.model, ChatOpenAI):
                return 'openai'
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            pass
        else:
            if isinstance(self.model, ChatOllama):
                return 'ollama'
        return 'generic'

    def _payload(self, messages, kwargs, kind):
        try:
            if kind == 'openai':
                payload = deepcopy(self.model._get_request_payload(messages, **kwargs))
                extra = payload.pop('extra_body', None) or {}
                if not isinstance(extra, Mapping):
                    raise self._error('extra_body must be a mapping')
                payload.update(extra)
                return payload
            if kind == 'ollama':
                return deepcopy(self.model._chat_params(messages, **deepcopy(kwargs)))
            defaults = {}
            for name in ('max_tokens', 'max_completion_tokens', 'max_output_tokens'):
                value = getattr(self.model, name, None)
                if value is not None:
                    defaults[name] = value
            return {'messages': messages, **defaults, **deepcopy(kwargs)}
        except ContextBudgetError:
            raise
        except Exception as exc:
            raise self._error('cannot serialize provider request') from exc

    def _tokenize(self, payload):
        if 'messages' not in payload:
            if self.counter == 'vllm':
                raise self._error('vllm counting requires Chat Completions routing')
            self._tokenizer_supported = False
            return None
        root_client = self.model.root_client
        parts = urlsplit(str(root_client.base_url))
        # Official OpenAI is not a vLLM discovery target.
        if self.counter == 'auto' and parts.hostname == 'api.openai.com':
            self._tokenizer_supported = False
            return None
        path = parts.path.rstrip('/')
        if path.endswith('/v1'):
            path = path[:-3]
        endpoint = urlunsplit((parts.scheme, parts.netloc, path + '/tokenize', '', ''))
        fields = (
            'model', 'messages', 'tools', 'add_generation_prompt',
            'continue_final_message', 'add_special_tokens', 'chat_template',
            'chat_template_kwargs', 'media_io_kwargs', 'mm_processor_kwargs',
        )
        body = {name: deepcopy(payload[name]) for name in fields if name in payload}
        template = dict(body.get('chat_template_kwargs') or {})
        for name in ('documents', 'reasoning_effort'):
            value = payload.get(name)
            if value is not None and value != 'auto':
                # Match ChatCompletionRequest.build_chat_params / merge_kwargs.
                template[name] = deepcopy(value)
        if template:
            body['chat_template_kwargs'] = template
        try:
            result = root_client.with_options(timeout=10, max_retries=0).post(
                endpoint, cast_to=Dict[str, Any], body=body,
                options={'follow_redirects': False,
                         **({'headers': payload['extra_headers']}
                            if payload.get('extra_headers') else {})})
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            if (self.counter == 'auto' and self._tokenizer_supported is None
                    and status in (404, 405, 501)):
                self._tokenizer_supported = False
                return None
            category = ('tokenizer authentication failed' if status in (401, 403)
                        else 'tokenizer endpoint unavailable')
            raise self._error(category, status_code=status) from exc
        if (not isinstance(result, Mapping)
                or not isinstance(result.get('count'), int)
                or isinstance(result.get('count'), bool)
                or result['count'] < 0
                or not _positive(result.get('max_model_len'))):
            raise self._error('invalid tokenizer response')
        self._tokenizer_supported = True
        self._serving_window = result['max_model_len']
        return result['count']

    def measure(self, messages, invoke_kwargs=None):
        """Count a complete request and return its per-invocation output cap."""
        messages = list(messages)
        original_kwargs = dict(invoke_kwargs or {})
        effective_kwargs = {**self.bound_kwargs, **original_kwargs}
        kind = self._kind()
        if not callable(self.counter) and self.counter not in ('auto', 'vllm', 'estimated'):
            raise self._error('counter must be auto, vllm, estimated, or callable')
        for name in ('context_window', 'output_tokens'):
            value = self.config.get(name)
            if value is not None and not _positive(value):
                raise self._error(f'{name} must be a positive integer')
        payload = self._payload(messages, effective_kwargs, kind)
        count = None
        method = 'estimated'
        if callable(self.counter):
            try:
                result = self.counter(deepcopy(payload))
            except Exception as exc:
                raise self._error('injected counter failed') from exc
            if isinstance(result, Mapping):
                count = result.get('count')
                window = result.get('max_model_len')
                if not _positive(window):
                    raise self._error('injected counter returned invalid capacity')
                self._serving_window = window
            else:
                count = result
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise self._error('injected counter returned invalid count')
            method = 'injected'
        elif self.counter == 'vllm' or (
                self.counter == 'auto' and kind == 'openai'
                and self._tokenizer_supported is not False):
            if kind != 'openai':
                raise self._error('vllm counting requires ChatOpenAI')
            count = self._tokenize(payload)
            if count is not None:
                method = 'vllm'
        if count is None:
            count = count_tokens_approximately(
                messages, tools=payload.get('tools'), chars_per_token=3,
                extra_tokens_per_message=12)
            # Template inputs are additional prompt data, not generation controls.
            import json
            extras = {name: payload[name] for name in
                      ('chat_template', 'chat_template_kwargs', 'documents')
                      if payload.get(name) is not None}
            if extras:
                count += math.ceil(len(json.dumps(extras, ensure_ascii=False)) / 3)

        profile = getattr(self.model, 'profile', None) or {}
        profile = profile if isinstance(profile, Mapping) else {}
        # A model profile is authoritative for the named OpenAI service, not for
        # a custom server merely reusing an OpenAI model name.
        if kind == 'openai' and urlsplit(str(self.model.root_client.base_url)).hostname != 'api.openai.com':
            profile = {}
        profile_input = profile.get('max_input_tokens')
        profile_output = profile.get('max_output_tokens')
        runtime_window = self._serving_window
        if kind == 'ollama':
            runtime_window = payload.get('options', {}).get('num_ctx')
        explicit_window = self.config.get('context_window')
        if runtime_window is not None and not _positive(runtime_window):
            raise self._error('provider context window must be a positive integer')
        if runtime_window and explicit_window and explicit_window > runtime_window:
            raise self._error('configured context window exceeds provider capacity')
        combined_window = explicit_window or runtime_window
        if not combined_window and not _positive(profile_input):
            raise self._error('unknown capacity; configure context_window')
        window = combined_window or profile_input

        if kind == 'ollama':
            existing_cap = payload.get('options', {}).get('num_predict')
        elif 'input' in payload and 'messages' not in payload:
            existing_cap = payload.get('max_output_tokens')
        else:
            existing_cap = payload.get('max_completion_tokens',
                                       payload.get('max_tokens', payload.get('max_output_tokens')))
        output = self.config.get('output_tokens')
        if output and _positive(existing_cap) and output > existing_cap:
            raise self._error('configured output_tokens exceeds existing output cap')
        if output is None:
            output = existing_cap if _positive(existing_cap) else min(8192, window // 4)
            if not _positive(existing_cap) and _positive(profile_output):
                output = min(output, profile_output)
        if not _positive(output):
            raise self._error('no positive generation allowance')
        if _positive(profile_output) and output > profile_output:
            raise self._error('output allowance exceeds provider output ceiling')
        headroom = (max(1024, math.ceil(window * .15)) if method == 'estimated' else 256)
        input_limit = window - headroom - (output if combined_window else 0)
        if _positive(profile_input):
            input_limit = min(input_limit, profile_input - headroom)
        if input_limit <= 0:
            raise self._error('output allowance and headroom exhaust context window',
                              context_window=window, output_tokens=output, headroom=headroom)

        invocation = deepcopy(original_kwargs)
        if kind == 'ollama':
            invocation['options'] = {**deepcopy(payload.get('options') or {}), 'num_predict': output}
            if runtime_window is None:
                invocation['options']['num_ctx'] = window
        elif kind == 'openai':
            # LangChain renames this for Responses routing. Providing only
            # max_output_tokens lets a configured legacy cap overwrite it.
            cap_key = 'max_completion_tokens'
            invocation.pop('max_tokens', None)
            invocation.pop('max_completion_tokens', None)
            invocation[cap_key] = output
            if 'max_tokens' in effective_kwargs:
                # RunnableBinding will merge its legacy key back otherwise.
                invocation['max_tokens'] = output
            if 'max_output_tokens' in effective_kwargs:
                invocation['max_output_tokens'] = output
            # extra_body is merged last by the SDK; keep a cap there in sync too.
            extra = deepcopy(effective_kwargs.get('extra_body', getattr(self.model, 'extra_body', None)) or {})
            if any(key in extra for key in ('max_tokens', 'max_completion_tokens', 'max_output_tokens')):
                for key in ('max_tokens', 'max_completion_tokens', 'max_output_tokens'):
                    extra.pop(key, None)
                invocation['extra_body'] = extra
        else:
            cap_key = ('max_output_tokens' if 'max_output_tokens' in payload
                       and 'max_tokens' not in payload else 'max_tokens')
            invocation[cap_key] = output
        diagnostics = dict(role=self.role, method=method, context_window=window,
                           input_tokens=count, output_tokens=output, headroom=headroom,
                           input_limit=input_limit)
        return BudgetMeasurement(count, window, output, headroom, input_limit,
                                 method, invocation, diagnostics)

    def prepare(self, messages, invoke_kwargs=None):
        messages = list(messages)
        measurement = self.measure(messages, invoke_kwargs)
        if not measurement.fits:
            raise ContextBudgetError('mandatory request exceeds available input budget',
                                     measurement.diagnostics)
        return PreparedRequest(messages, measurement.invoke_kwargs, measurement.diagnostics)


def is_context_overflow(error):
    """Prefer provider error codes; use text only when no specific code exists.

Some local OpenAI-compatible servers expose only an error message, so the
bounded patterns remain a compatibility fallback. An explicit unrelated code
or non-capacity HTTP status must not trigger context shrinking and a retry.
"""
    status = getattr(error, 'status_code', None)
    if status is not None and status not in (400, 413, 422):
        return False
    code = getattr(error, 'code', None)
    body = getattr(error, 'body', None)
    # vLLM may repeat the HTTP status as its code; prefer a nested semantic code.
    if ((code is None or type(code) is int and code in (400, 413, 422))
            and isinstance(body, Mapping)):
        detail = body.get('error', body)
        if isinstance(detail, Mapping):
            code = detail.get('code')
    if type(code) is int and code in (400, 413, 422):
        code = None
    if code is not None:
        return isinstance(code, str) and code in {
            'context_length_exceeded', 'context_window_exceeded',
            'max_context_length_exceeded'}
    text = str(error).lower()
    return bool(re.search(
        r'context[_ ]length[_ ]exceeded|maximum context length|'
        r'(?:input|prompt).{0,100}(?:exceeds?|too long).{0,100}(?:token|context)|'
        r'(?:input|prompt).{0,100}(?:token).{0,100}(?:maximum|limit)|'
        r'requested.{0,50}tokens.{0,100}(?:maximum|context)', text))
