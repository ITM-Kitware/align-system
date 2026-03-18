from outlines.backends.outlines_core import OutlinesCoreBackend
from outlines.models.transformers import TransformerTokenizer
from outlines_core import Vocabulary


# monkey patch to fix https://github.com/dottxt-ai/outlines/pull/1831
# fix was applied to outlines main, but we will probably be blocked from updating due to vllm dependency
@staticmethod
def deterministic_create_vocab(vocab, eos_token_id, eos_token, token_to_str):
    formatted_vocab = {}
    for token, token_id in vocab.items():
        token_as_str = token_to_str(token)
        formatted_vocab.setdefault(token_as_str, []).append(token_id)
    formatted_vocab.pop(eos_token)
    return Vocabulary(eos_token_id, formatted_vocab)


OutlinesCoreBackend.create_outlines_core_vocabulary = deterministic_create_vocab


# monkey patch to fix https://github.com/dottxt-ai/outlines/pull/1817
# newer verion of outlines fixes this issue, but we are blocked with the vllm dependency
def convert_token_to_string(self, token: str) -> str:
    from transformers.file_utils import SPIECE_UNDERLINE

    string = self.tokenizer.convert_tokens_to_string([token])

    if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
        return " " + string

    return string


TransformerTokenizer.convert_token_to_string = convert_token_to_string
