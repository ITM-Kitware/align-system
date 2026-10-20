"""Chat Completions adapter that budgets history using vLLM's tokenizer."""

import asyncio

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openai import APIStatusError
from pydantic import Field

from align_system.exceptions import ContextWindowError
from align_system.utils import logging


log = logging.getLogger(__name__)


class VLLMChatOpenAI(ChatOpenAI):
    """Reserve generation space and evict old, complete tool exchanges.

    Requires the server's /tokenize endpoint. System messages, the opening
    user message, and the newest exchange are retained. The caller's full
    transcript is never modified. Streaming is disabled so every request
    passes through the same preflight check, including structured batches.
    """

    disable_streaming: bool = True
    context_safety_margin: int = Field(default=256, ge=0)

    def _count_prompt(self, payload):
        # ChatOpenAI's serializer is also used for the actual completion,
        # including tool schemas. extra_body overrides SDK request fields.
        payload = {**payload, **(payload.get('extra_body') or {})}
        if 'messages' not in payload:
            raise ContextWindowError(
                'VLLMChatOpenAI requires Chat Completions, not Responses API.')
        output_tokens = payload.get('max_completion_tokens')
        if output_tokens is None:
            output_tokens = payload.get('max_tokens')
        if type(output_tokens) is not int or output_tokens <= 0:
            raise ContextWindowError(
                'Set driver.chat_model.max_tokens to a positive response limit.')

        fields = ('model', 'messages', 'tools', 'chat_template',
                  'chat_template_kwargs', 'add_generation_prompt',
                  'continue_final_message', 'add_special_tokens',
                  'media_io_kwargs', 'mm_processor_kwargs')
        body = {key: payload[key] for key in fields if key in payload}
        url = str(self.root_client.base_url).rstrip('/').removesuffix('/v1')
        try:
            result = self.root_client.with_options(timeout=30, max_retries=0).post(
                url + '/tokenize', body=body, cast_to=dict[str, object])
        except APIStatusError as exc:
            if exc.status_code not in (400, 404, 422):
                raise
            raise ContextWindowError(
                f'vLLM /tokenize returned HTTP {exc.status_code}; token budgeting '
                'requires a reachable /tokenize endpoint accepting chat messages '
                'and tool schemas. Check the vLLM URL and server version.') from exc

        count, limit = result.get('count'), result.get('max_model_len')
        if (type(count) is not int or count < 0
                or type(limit) is not int or limit <= 0):
            raise ContextWindowError(
                'vLLM /tokenize did not return valid count and max_model_len values.')
        return count, limit, output_tokens

    def _prepare_messages(self, messages, stop=None, *, preserve_messages=False, **kwargs):
        window = list(messages)
        # Keep the leading system/developer instructions and opening user turn.
        pinned = 0
        while pinned < len(window) and window[pinned].type in ('system', 'developer'):
            pinned += 1
        if pinned < len(window) and isinstance(window[pinned], HumanMessage):
            pinned += 1

        exchanges = []
        for message in window[pinned:]:
            if isinstance(message, ToolMessage) and exchanges:
                exchanges[-1].append(message)
            else:
                exchanges.append([message])

        while True:
            payload = self._get_request_payload(window, stop=stop, **kwargs)
            count, limit, output_tokens = self._count_prompt(payload)
            budget = limit - output_tokens - self.context_safety_margin
            if count <= budget:
                removed = len(messages) - len(window)
                if removed:
                    log.info(
                        f'Qwen context: trimmed {removed} old messages; '
                        f'{count} prompt tokens + {output_tokens} output tokens '
                        f'+ {self.context_safety_margin} safety tokens <= {limit}.')
                return window
            if preserve_messages:
                raise ContextWindowError(
                    f'Qwen context cannot fit all required messages: '
                    f'{count} prompt tokens + {output_tokens} output tokens '
                    f'+ {self.context_safety_margin} safety tokens > server limit '
                    f'{limit}. Increase VLLM_MAX_MODEL_LEN when starting the server '
                    'or reduce the scoring output budget. No messages were trimmed '
                    'and no completion was sent.')
            if len(exchanges) <= 1:
                raise ContextWindowError(
                    f'Qwen context cannot fit the retained instructions and newest '
                    f'exchange: {count} prompt tokens + {output_tokens} output tokens '
                    f'+ {self.context_safety_margin} safety tokens > server limit '
                    f'{limit}. Increase VLLM_MAX_MODEL_LEN when starting the server, '
                    'reduce driver.chat_model.max_tokens, or shorten the current '
                    'observation/tool result. No completion was sent.')
            exchanges.pop(0)
            window = [*messages[:pinned],
                      *(message for exchange in exchanges for message in exchange)]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        preserve_messages = kwargs.pop('preserve_messages', False)
        window = self._prepare_messages(
            messages, stop=stop, preserve_messages=preserve_messages, **kwargs)
        return super()._generate(window, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        preserve_messages = kwargs.pop('preserve_messages', False)
        window = await asyncio.to_thread(
            self._prepare_messages, messages, stop=stop,
            preserve_messages=preserve_messages, **kwargs)
        return await super()._agenerate(
            window, stop=stop, run_manager=run_manager, **kwargs)
