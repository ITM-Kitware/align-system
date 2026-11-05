"""Optional ChatOpenAI integration retaining local reasoning extensions."""

from collections.abc import Mapping

from langchain_openai import ChatOpenAI

from .chat_transport import normalize_chat_response


class CompatibleChatOpenAI(ChatOpenAI):
    """Keep vLLM reasoning out of visible text while preserving its metadata.

    The agent uses nonstreaming invocations. Standard ChatOpenAI currently drops
    the nonstandard reasoning fields during Chat Completions deserialization.
    """

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        data = (response if isinstance(response, Mapping) else response.model_dump(
            exclude={'choices': {'__all__': {'message': {'parsed'}}}}, warnings=False))
        for generation, choice in zip(result.generations, data.get('choices', [])):
            raw = choice.get('message', {})
            for name in ('reasoning_content', 'reasoning'):
                value = raw.get(name)
                if isinstance(value, str):
                    generation.message.additional_kwargs[name] = value
            generation.message = normalize_chat_response(generation.message)
        return result
