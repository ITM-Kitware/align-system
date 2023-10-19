import pytest

from align_system.language_model_lib.chat_language_model import ChatLanguageModel

MODEL_TO_TEST = 'meta-llama/Llama-2-7b-chat-hf'

@pytest.fixture(scope="module")
def chat_language_model():
    # Load the model once for all tests that use this fixture
    return ChatLanguageModel.load_model(MODEL_TO_TEST)


def test_generate_responses(chat_language_model):
    dialogs = [
        [
            {'role': 'system', 'content': 'speak like a pirate'},
            {'role': 'user', 'content': 'hello'},
        ],
        [
            {'role': 'system', 'content': 'speak like a pirate'},
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'What if you'},
        ],
        [
            {'role': 'system', 'content': 'speak like a pirate'},
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'What if you'},
        ]
    ]
    
    responses = chat_language_model.generate_responses(dialogs, max_new_tokens=512, temperature=0.0001)
    
    assert type(responses) is list
    assert len(responses) == len(responses)
    assert type(responses[0]) is str
    assert responses[1].startswith(dialogs[1][-1]['content'])
    


