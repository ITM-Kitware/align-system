"""Offline contracts for the legacy and upgraded local inference runtimes."""

from importlib import import_module
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


ALGORITHMS_INIT = Path(__file__).resolve().parents[1] / 'algorithms' / '__init__.py'


@pytest.mark.parametrize('target', [
    ('outlines_inference_engine', 'OutlinesTransformersInferenceEngine'),
    ('outlines_adm', 'OutlinesTransformersADM'),
    ('outlines_regression_adm_comparative', 'OutlinesTransformersComparativeRegressionADM'),
])
@pytest.mark.parametrize('automatic_template', [False, True])
def test_outlines_constructors_apply_the_chat_template_once(
        monkeypatch, target, automatic_template):
    module_name, class_name = target
    module = import_module('align_system.algorithms.' + module_name)
    template_calls = []

    class Tokenizer:
        def apply_chat_template(self, dialog):
            template_calls.append(dialog)
            return '<user>' + dialog[0]['content'] + '</user><assistant>'

        def decode(self, value):
            return value

    class Adapter:
        def format_input(self, prompt):
            if getattr(self, 'has_chat_template', False):
                return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}])
            return prompt

    tokenizer = Tokenizer()
    adapter = Adapter()
    if automatic_template:
        adapter.has_chat_template = True
    wrapped = SimpleNamespace(
        tokenizer=SimpleNamespace(tokenizer=tokenizer), type_adapter=adapter)
    model = object()
    monkeypatch.setattr(module.transformers.AutoModelForCausalLM, 'from_pretrained',
                        lambda *args, **kwargs: model)
    monkeypatch.setattr(module.transformers.AutoTokenizer, 'from_pretrained',
                        lambda *args, **kwargs: tokenizer)

    def wrap(actual_model, actual_tokenizer, **kwargs):
        assert actual_model is model
        assert actual_tokenizer is tokenizer
        return wrapped

    monkeypatch.setattr(module.outlines, 'from_transformers', wrap)
    engine = getattr(module, class_name)('offline-model', precision='full')
    prompt = engine.dialog_to_prompt([{'role': 'user', 'content': 'hello'}])

    assert engine.model.type_adapter.format_input(prompt) == '<user>hello</user><assistant>'
    assert len(template_calls) == 1
    assert not hasattr(engine.model.tokenizer, 'is_llama')


def test_modern_outlines_keeps_upstream_tokenizer_and_vocabulary_fixes(monkeypatch):
    from outlines.backends.outlines_core import OutlinesCoreBackend
    from outlines.models.transformers import TransformerTokenizer

    original_vocab = OutlinesCoreBackend.create_outlines_core_vocabulary
    original_token = TransformerTokenizer.convert_token_to_string
    monkeypatch.setattr('importlib.metadata.version', lambda name: '1.3.3')

    runpy.run_path(str(ALGORITHMS_INIT))

    assert OutlinesCoreBackend.create_outlines_core_vocabulary is original_vocab
    assert TransformerTokenizer.convert_token_to_string is original_token


def test_legacy_outlines_preserves_spacing_and_duplicate_token_order(monkeypatch):
    import outlines_core
    from outlines.backends.outlines_core import OutlinesCoreBackend
    from outlines.models.transformers import TransformerTokenizer

    # Register the current attributes for restoration after executing the legacy
    # package initialization in a separate namespace.
    monkeypatch.setattr(OutlinesCoreBackend, 'create_outlines_core_vocabulary',
                        OutlinesCoreBackend.create_outlines_core_vocabulary)
    monkeypatch.setattr(TransformerTokenizer, 'convert_token_to_string',
                        TransformerTokenizer.convert_token_to_string)
    monkeypatch.setattr('importlib.metadata.version', lambda name: '1.2.9')
    monkeypatch.setattr(outlines_core, 'Vocabulary', lambda eos, vocab: (eos, vocab))
    runpy.run_path(str(ALGORITHMS_INIT))

    tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(
        convert_tokens_to_string=lambda tokens: tokens[0].removeprefix('\u2581')))
    assert TransformerTokenizer.convert_token_to_string(tokenizer, '\u2581café') == ' café'
    assert TransformerTokenizer.convert_token_to_string(tokenizer, 'café') == 'café'
    eos, vocabulary = OutlinesCoreBackend.create_outlines_core_vocabulary(
        {'a': 7, 'alias': 2, '<eos>': 9}, 9, '<eos>',
        lambda token: 'a' if token == 'alias' else token)
    assert eos == 9
    assert vocabulary == {'a': [7, 2]}


def test_outlines_generates_single_and_batch_json_with_a_local_tiny_model():
    import outlines
    from outlines.types import JsonSchema
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from align_system.algorithms.outlines_compat import from_transformers

    vocabulary = {'<unk>': 0, '<pad>': 1, '<eos>': 2, 'hello': 3, 'true': 4, 'false': 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='<unk>', pad_token='<pad>', eos_token='<eos>')
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] }}"
        "{% endfor %}assistant: ")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(vocabulary), n_positions=32, n_embd=16, n_layer=1, n_head=1,
            bos_token_id=2, eos_token_id=2, pad_token_id=1)).eval()
        wrapped = from_transformers(model, tokenizer)
        assert wrapped.type_adapter.format_input('hello') == 'hello'
        generator = outlines.Generator(wrapped, JsonSchema('{"type":"boolean"}'))
        single = generator('hello', max_new_tokens=4, do_sample=False)
        batch = generator.batch(['hello', 'hello'], max_new_tokens=4, do_sample=False)

    assert type(json.loads(single)) is bool
    assert len(batch) == 2
    assert all(type(json.loads(output)) is bool for output in batch)


@pytest.mark.parametrize('transformers_version', ['4.57.6', '5.19.0'])
@pytest.mark.parametrize('device_count', [0, 1, 2])
def test_kaleido_loading_preserves_single_device_and_supports_modern_multi_gpu(
        monkeypatch, transformers_version, device_count):
    from align_system.algorithms.lib import kaleido

    calls = []
    device = 'cpu' if device_count == 0 else 'cuda:0'

    class Model:
        config = SimpleNamespace(task_specific_params={
            task: {'template': '{{input}}', 'args': {}}
            for task in ('generate', 'relevance', 'valence', 'explanation')})

        def __init__(self):
            self.device = device

        def parallelize(self):
            calls.append('parallelize')

        def to(self, target):
            calls.append(('to', target))
            return self

        def eval(self):
            calls.append('eval')

    def load(name, **kwargs):
        calls.append(('load', kwargs))
        return Model()

    monkeypatch.setattr(kaleido, 'version', lambda name: transformers_version)
    monkeypatch.setattr(kaleido.torch.cuda, 'device_count', lambda: device_count)
    monkeypatch.setattr(kaleido.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(kaleido.AutoModelForSeq2SeqLM, 'from_pretrained', load)
    monkeypatch.setattr(kaleido.AutoTokenizer, 'from_pretrained',
                        lambda name: lambda text: SimpleNamespace(input_ids=[1]))
    system = kaleido.KaleidoSys(device=device)
    system.load_model()

    automatic_map = device_count > 1 and transformers_version.startswith('5.')
    assert calls[0] == ('load', {'device_map': 'auto'} if automatic_map else {})
    assert ('parallelize' in calls) == (device_count > 1 and not automatic_map)
    assert (('to', device) in calls) == (device_count <= 1)
    assert system.device == device
    assert 'eval' in calls


@pytest.mark.parametrize('prompts', ['hello', ['hello', 'goodbye']])
@pytest.mark.parametrize('schema, constraint', [
    ('{"type":"object","properties":{"ok":{"type":"boolean"}}}', 'json'),
    (r'\{"ok":true\}', 'regex'),
])
def test_in_process_vllm_keeps_structured_single_and_batch_contracts(
        monkeypatch, prompts, schema, constraint):
    module = pytest.importorskip('align_system.algorithms.vllm_inference_engine')
    calls = []

    class LLM:
        def __init__(self, model):
            assert model == 'offline-model'

        def generate(self, prompts, sampling_params):
            calls.append(sampling_params)
            size = 1 if isinstance(prompts, str) else len(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(text='{"ok":true}')])
                    for _ in range(size)]

    monkeypatch.setattr(module, 'LLM', LLM)
    engine = module.VLLMInferenceEngine('offline-model', {'temperature': 0.0})
    result = engine.run_inference(prompts, schema)

    expected = {'ok': True} if isinstance(prompts, str) else [{'ok': True}, {'ok': True}]
    assert result == expected
    assert calls[0].max_tokens == 8192
    assert getattr(calls[0].structured_outputs, constraint) is not None


def test_runtime_markers_keep_a_python310_fallback():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / 'pyproject.toml').read_text())
    expected = {
        '3.10': {'vllm': '0.19.0', 'outlines': '1.2.9', 'transformers': '4.57.6'},
        '3.11': {'vllm': '0.32.0', 'outlines': '1.3.3', 'transformers': '5.19.0'},
        '3.12': {'vllm': '0.32.0', 'outlines': '1.3.3', 'transformers': '5.19.0'},
    }
    requirements = [Requirement(item) for item in project['project']['dependencies']]
    for python, versions in expected.items():
        environment = {**default_environment(), 'python_version': python,
                       'python_full_version': python + '.0'}
        selected = [r for r in requirements if not r.marker or r.marker.evaluate(environment)]
        for name, version in versions.items():
            matches = [r for r in selected if r.name == name]
            assert len(matches) == 1
            assert version in matches[0].specifier
