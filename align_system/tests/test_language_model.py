import pytest
import torch

from align_system.algorithms.lib.language_model import LanguageModel

MODEL_TO_TEST = 'gpt2'  # Use a smaller model for testing

@pytest.fixture(scope="module")
def language_model():
    # Load the model once for all tests that use this fixture
    return LanguageModel.load_model(MODEL_TO_TEST, device='cpu')

def test_load_model(language_model):
    assert language_model.model.dtype == torch.float32
    assert language_model.model.device.type == 'cpu'


def test_generate_from_tokens(language_model):
    tokens = [
        [9246, 9703, 9246, 9703],
        [1681, 146, 1681, 146, 1681],
    ]
    
    generations = language_model.generate_from_tokens(tokens, max_new_tokens=1, temperature=0)
    
    assert generations == [
        'cat',
        '\n'
    ]

def test_generate(language_model):
    prompts = [
        'catdogcatdog',
        'ABCABCABCABCABC',
    ]
    generations = language_model.generate(prompts, max_new_tokens=1, temperature=0)
    assert generations == [
        'cat',
        'ABC',
    ]

def test_generate_with_prefixes(language_model):
    prompts = [
        'catdogcatdog',
        'ABCABCABCABCABC',
    ]
    prefixes = [
        'cat',
        'ABC',
    ]
    generations = language_model.generate_with_prefixes(prompts, prefixes=prefixes, max_new_tokens=1, temperature=0)
    
    for generation, prefix in zip(generations, prefixes):
        assert generation.startswith(prefix)