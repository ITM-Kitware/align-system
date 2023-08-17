import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json


def find_sequence(arr, seq):
    seq_len = len(seq)
    matches = [i+seq_len for i in range(len(arr)) if list(arr[i:i+seq_len]) == list(seq)]
    if matches:
        return matches[0]
    else:
        return None


def get_logits(output_scores, start_idx, letter_ids):
    logits = []
    for _, id_ in letter_ids.items():
        logit = output_scores[start_idx][0, id_].item()
        logits.append(logit)
    return logits


def to_probabilities(logits):
    return torch.nn.functional.softmax(torch.tensor(logits), dim=0)  # Apply softmax


class LLMChatBaseline:
    
    def __init__(self, device='cuda', hf_model='meta-llama/Llama-2-7b-chat-hf'):
        self.device = device
        self.hf_model = hf_model
        
        self.model = None
        self.tokenizer = None
    
    
    def load_model(self):
        print('Loading model:', self.hf_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_model, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        
        self.model = self.model.to(self.device)
    
    
    def get_character_ids(self, character_str):
        assert 'llama-2' in self.hf_model.lower(), "This function is only compatible with llama-2 models."
        assert list(character_str) == ['0', '1', '2', '3'], "character_str must be a string of the characters '0', '1', '2', '3'."
        return {
            '0': 29900,
            '1': 29896,
            '2': 29906,
            '3': 29941,
        } # TODO use the tokenizer to find the ids
    
    
    def get_search_sequence(self):
        assert 'llama-2' in self.hf_model.lower(), "This function is only compatible with llama-2 models."
        return [22550, 1115, 29871] # TODO use the tokenizer to calculate this
    
    
    def chat_prompt_tokens(self, dialogs, return_tensor=True):
        # Define instance and system borders
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        # Initialize an empty list to hold prompt tokens
        prompt_tokens = []

        # Iterate through each dialog in dialogs
        for dialog in dialogs:
            # If the role of the first message is system
            if dialog[0]["role"] == "system":
                # Create an initial dialog entry combining system and user messages
                system_dialog = {"role": dialog[1]["role"], 
                                "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"]}
                # Update dialog to start with system_dialog and followed by the rest of the dialog
                dialog = [system_dialog] + dialog[2:]
                
            # Ensure the correct dialog order (system, user, assistant, user, assistant... )
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]), \
                "Model only supports 'system', 'user' and 'assistant' roles, in the sequence (s/u/a/u/a...)"
            
            # Encode each user message and its following assistant message into tokens
            dialog_tokens = []
            for prompt, answer in zip(dialog[::2], dialog[1::2]):
                tokenized_message = ([self.tokenizer.bos_token_id] + 
                                    self.tokenizer.encode(f"{B_INST} {prompt['content'].strip()} {E_INST} {answer['content'].strip()} ", 
                                                    add_special_tokens=False) + 
                                    [self.tokenizer.eos_token_id])
                dialog_tokens.extend(tokenized_message)

            # Ensure the final message is from the user
            assert dialog[-1]["role"] == "user", "Last message must be from the user."

            # Encode the user's final message into tokens and add to dialog_tokens
            user_final_message_tokens = ([self.tokenizer.bos_token_id] + self.tokenizer.encode(
                f"{B_INST} {dialog[-1]['content'].strip()} {E_INST}",
                add_special_tokens=False))
            dialog_tokens.extend(user_final_message_tokens)

            # Add the final tokenized dialog to prompt_tokens
            prompt_tokens.append(dialog_tokens)

        # Convert list of tokens to tensor if specified 
        if return_tensor:
            prompt_tokens = torch.tensor(prompt_tokens)
        
        # Return final list or tensor of prompt tokens
        return prompt_tokens
    
    
    
    def answer_multiple_choice(self, question, options, system_message_id=0, verbose=True):
        medical_triage_system_message = (
            "You are an assistant specialized in answering multiple-choice questions related to "
            "medical triage. When a question is posed, carefully analyze the symptoms or conditions "
            "described. Respond by providing a detailed reasoning using a step-by-step process or "
            "process of elimination. Conclude with the final answer, represented by the corresponding "
            "index number. Your response should be grounded in established medical knowledge and aim "
            "to be informative. Please format your response as a JSON object in the following structure:\\n\\n"
            "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>}\\n"
            "Ensure that you adhere to proper JSON syntax, and carefully note the index that corresponds to each answer."
        )

        system_messages = [
            medical_triage_system_message
            # "You are an assistant capable of providing answers to multiple choice medical-related questions. When a question is asked, analyze the symptoms or conditions mentioned. Respond by providing your reasoning in a train-of-thought/process of elimination manner. Afterwards, provide the final answer in the form of the designated index. Your response should be factual based on established medical knowledge and aim to be helpful. However, format your response as a JSON object, containing a string and an integer, of the form:\n\n{\"Reasoning\": \"<Your reasoned explanation>\", \"Answer\": <Integer index of your final answer>}\n\nUse proper JSON syntax and pay close attention to which index corresponds to which answer."
        ]
        
        formatted_options = [f'({i}) {option}' for i, option in enumerate(options)]
        
        content = f'{question} {formatted_options}'
        
        dialog = [
            {
                "role": "system",
                "content": system_messages[system_message_id]
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        prompt_tokens = self.chat_prompt_tokens([dialog], return_tensor=False)
        
        
        prompt_length = len(prompt_tokens[0])
        
        prefix_tokens = self.tokenizer.encode('{"Reasoning": "', add_special_tokens=False) # TODO make this connected to the system message
        prompt_tokens[0] += prefix_tokens
        
        prompt_tokens = torch.tensor(prompt_tokens)
        prompt_tokens = prompt_tokens.to(self.device)
                
        outputs = self.model.generate(prompt_tokens, return_dict_in_generate=True, output_scores=True, max_new_tokens=512)

        # Print the generated model output
        generated_output = self.tokenizer.decode(outputs.sequences[0][prompt_length:])
        
        # try to parse out the reasoning string
        reasoning = None
        answer_idx = None
        try:
            start_idx = generated_output.find('{')
            end_idx = generated_output.rfind('}')
            json_str = generated_output[start_idx:end_idx+1]
            generated_data = json.loads(json_str)
            
            try:
                reasoning = generated_data['Reasoning']
            except KeyError:
                if verbose:
                    print('Warning: could not parse reasoning from generated output. Missing key: "Reasoning"')
            
            try:
                answer_idx = generated_data['Answer']
            except KeyError:
                if verbose:
                    print('Warning: could not parse answer index from generated output. Missing key: "Answer"')
        except Exception as e:
            if verbose:
                print('Warning: could not parse reasoning or answer index from generated output.')

        # Find sequence
        output_ids = np.array(outputs.sequences[0].cpu())[len(prompt_tokens[0]):]
        search_sequence = self.get_search_sequence()
        start_idx = find_sequence(output_ids, search_sequence)

        if start_idx is not None:
            # Get answer letters and logits
            answer_character_ids = self.get_character_ids('0123')
            logits = get_logits(outputs.scores, start_idx, answer_character_ids)

            # Print probabilities
            probabilities = to_probabilities(logits)
            return generated_output, reasoning, answer_idx, probabilities.tolist()
        else:
            if verbose:
                print('Warning: could not find search sequence in generated output and was unable to calculate probabilities.')
            return generated_output, reasoning, answer_idx, None