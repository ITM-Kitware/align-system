from importlib.metadata import version


if version('outlines') == '1.2.9':
    from outlines.backends.outlines_core import OutlinesCoreBackend
    from outlines.models.transformers import TransformerTokenizer
    from outlines_core import Vocabulary

    # Retain these fixes for the Python 3.10 runtime. Outlines 1.3.3 includes
    # both upstream fixes, including additional EOS handling we must not replace.
    # https://github.com/dottxt-ai/outlines/pull/1831
    @staticmethod
    def deterministic_create_vocab(vocab, eos_token_id, eos_token, token_to_str):
        formatted_vocab = {}
        for token, token_id in vocab.items():
            token_as_str = token_to_str(token)
            formatted_vocab.setdefault(token_as_str, []).append(token_id)
        formatted_vocab.pop(eos_token)
        return Vocabulary(eos_token_id, formatted_vocab)

    OutlinesCoreBackend.create_outlines_core_vocabulary = deterministic_create_vocab

    # https://github.com/dottxt-ai/outlines/pull/1817
    def convert_token_to_string(self, token: str) -> str:
        string = self.tokenizer.convert_tokens_to_string([token])

        if token.startswith('\u2581') or token == "<0x20>":
            return " " + string

        return string

    TransformerTokenizer.convert_token_to_string = convert_token_to_string
