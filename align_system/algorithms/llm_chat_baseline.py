import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import re


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


STANDARD_MULTIPLE_CHOICE_JSON_FORMAT = "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>}\\n"

TREATMENT_MULTIPLE_CHOICE_JSON_FORMAT = "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>, \"Location\": \"<Specific location on the body where the treatment should be applied, one of: 'right forearm', 'left forearm', 'right calf', 'left calf', 'right thigh', 'left thigh', 'right stomach', 'left stomach', 'right bicep', 'left bicep', 'right shoulder', 'left shoulder', 'right side', 'left side', 'right chest', 'left chest', 'right wrist', 'left wrist', 'left face', 'right face', 'left neck', 'right neck', 'unspecified'>\"}\\n"

TAGGING_MULTIPLE_CHOICE_JSON_FORMAT = "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>, \"Tag\": \"<Specific medical triage tag to apply, one of: 'MINIMAL', 'DELAYED', 'IMMEDIATE', 'EXPECTANT'>\"}\\n"


class LLMChatBaseline:

    def __init__(self, device='cuda', hf_model='meta-llama/Llama-2-7b-chat-hf', precision='full', temperature=0.7):
        self.device = device
        self.hf_model = hf_model
        self.temperature = temperature

        assert precision in ['full', 'half'], "precision must be either 'full' or 'half'."
        self.precision = torch.float32 if precision == 'full' else torch.float16

        self.model = None
        self.tokenizer = None


    def load_model(self):
        print('Loading model:', self.hf_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_model, torch_dtype=self.precision)
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


    def answer_multiple_choice(self, question, options, system_message=None, prefix=None, verbose=True, json_format=STANDARD_MULTIPLE_CHOICE_JSON_FORMAT):
        medical_triage_system_message = (
            "You are an assistant specialized in answering multiple-choice questions related to "
            "medical triage. When a question is posed, carefully analyze the symptoms or conditions "
            "described. Respond by providing a detailed reasoning using a step-by-step process or "
            "process of elimination. Conclude with the final answer, represented by the corresponding "
            "index number. Your response should be grounded in established medical knowledge and aim "
            "to be informative. Please format your response as a JSON object in the following structure:\\n\\n"
            f"{json_format}"
            "Ensure that you adhere to proper JSON syntax, and carefully note the index that corresponds to each answer."
        )
        if system_message is None:
            system_message = medical_triage_system_message
            if prefix is None:
                prefix = '{"Reasoning": "'

        formatted_options = [f'({i}) {option}' for i, option in enumerate(options)]

        content = f'{question} {formatted_options}'

        dialog = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": content
            }
        ]

        prompt_tokens = self.chat_prompt_tokens([dialog], return_tensor=False)


        prompt_length = len(prompt_tokens[0])

        if prefix is not None:
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            prompt_tokens[0] += prefix_tokens

        prompt_tokens = torch.tensor(prompt_tokens)
        prompt_tokens = prompt_tokens.to(self.device)

        outputs = self.model.generate(prompt_tokens, return_dict_in_generate=True, output_scores=True, max_new_tokens=512, temperature=self.temperature)

        # Print the generated model output
        generated_output = self.tokenizer.decode(outputs.sequences[0][prompt_length:])

        return generated_output

        # # try to parse out the reasoning string
        # reasoning = None
        # answer_idx = None
        # try:
        #     try:
        #         start_idx = generated_output.find('{')
        #         end_idx = generated_output.rfind('}')
        #         json_str = generated_output[start_idx:end_idx+1]
        #         generated_data = json.loads(json_str)
        #     except Exception:
        #         generated_data = self.correct_json(generated_output, verbose=verbose)

        #     try:
        #         reasoning = generated_data['Reasoning']
        #     except KeyError:
        #         if verbose:
        #             print('Warning: could not parse reasoning from generated output. Missing key: "Reasoning"')

        #     try:
        #         answer_idx = generated_data['Answer']
        #     except KeyError:
        #         if verbose:
        #             print('Warning: could not parse answer index from generated output. Missing key: "Answer"')
        # except Exception as e:
        #     if verbose:
        #         print('Warning: could not parse reasoning or answer index from generated output.')

        # # Find sequence
        # output_ids = np.array(outputs.sequences[0].cpu())[len(prompt_tokens[0]):]
        # search_sequence = self.get_search_sequence()
        # start_idx = find_sequence(output_ids, search_sequence)

        # if start_idx is not None:
        #     # Get answer letters and logits
        #     answer_character_ids = self.get_character_ids('0123')
        #     logits = get_logits(outputs.scores, start_idx, answer_character_ids)

        #     # Print probabilities
        #     probabilities = to_probabilities(logits)
        #     return generated_output, reasoning, answer_idx, probabilities.tolist()
        # else:
        #     if verbose:
        #         print('Warning: could not find search sequence in generated output and was unable to calculate probabilities.')
        #     return generated_output, reasoning, answer_idx, None

    @staticmethod
    def parse_generated_output(generated_output):

        # initialize variables
        reasoning = None
        answer_idx = None

        # Remove trailing characters
        output = generated_output.replace('</s>', '')
        end_idx = output.rfind('}')+1
        start_id = output.find('{')
        if end_idx != -1:
            output = output[:end_idx]
        if start_id != -1:
            output = output[start_id:]

        # Replace in-line newlines
        output = re.sub(r'\n', ' ', output)

        # Fix missing commas
        output = re.sub(r'"\s+"', '", "', output)

        # Parse json output
        try:
            parsed = json.loads(output)
            if 'Reasoning' in parsed:
                reasoning = parsed['Reasoning']

            if 'Answer' in parsed:
                answer_idx = parsed['Answer']

        except json.JSONDecodeError:
            pass

        if answer_idx is None:
            # If json parsing fails, do string parsing
            start_idx = generated_output.find('"Reasoning":')
            end_idx = generated_output.find('",', start_idx)
            if start_idx != -1 and end_idx != -1:
                reasoning = generated_output[start_idx + len('"Reasoning":'):end_idx]

            search_strings = ['Answer":', 'Answer:', 'Answer\\":', 'answer is', 'index']
            for string in search_strings:
                # try to parse the string "Answer": ... ",
                start_idx = generated_output.lower().rfind(string.lower())
                if start_idx != -1:
                    # find the next numeric character
                    chars = generated_output[start_idx + len(string):]
                    for char in chars:
                        if char.isnumeric():
                            answer_idx = int(char)
                            break

                if answer_idx is not None:
                    break

        return reasoning, answer_idx

    @staticmethod
    def attempt_generic_parse(generated_output, fields_of_interest):
        # Remove trailing characters
        output = generated_output.replace('</s>', '')
        end_idx = output.rfind('}')+1
        start_id = output.find('{')
        if end_idx != -1:
            output = output[:end_idx]
        if start_id != -1:
            output = output[start_id:]

        # Replace in-line newlines
        output = re.sub(r'\n', ' ', output)

        # Fix missing commas
        output = re.sub(r'"\s+"', '", "', output)

        # Parse json output
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            pass
        else:
            try:
                return {f: parsed[f] for f in fields_of_interest}
            except KeyError:
                pass

        parsed_output = {}
        for field in fields_of_interest:
            parsed_field = None
            if m := re.search(rf'"{field}"\s*:\s*"([^"]*)"', output):  # noqa
                parsed_field = m.group(1)
            elif m := re.search(rf'"{field}"'+'\s*:\s*([^\s,}]*)', output):  # noqa
                parsed_field = m.group(1)
            elif m := re.search(rf'{field}'+'\s*:\s*([^\s,}]*)', output):  # noqa
                parsed_field = m.group(1)

            # Failed to parse every field
            if parsed_field is None:
                return None
            else:
                # Special handling of common "Index" field (should be
                # an integer)
                if field == 'Index':
                    if m := re.search(r'\d+', parsed_field):  # noqa
                        parsed_field = m.group(0)

                    try:
                        parsed_field = int(parsed_field)
                    except ValueError:
                        # Failed to parse
                        return None

            parsed_output[field] = parsed_field

        return parsed_output

    def correct_json(self, invalid_json, verbose=True):
        # Custom system message for correcting invalid JSON
        system_message = (
            "You are an assistant specialized in correcting malformed JSON strings. "
            "Analyze the provided JSON string and correct any syntactical errors "
            "to make it a valid JSON object. Ensure that your corrections adhere "
            "to proper JSON syntax."
            "Do not provide an explanation or output any text other than the corrected JSON object."
        )

        # Dialog with the system message and the invalid JSON
        dialog = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": invalid_json
            }
        ]

        # Generate the prompt tokens similarly to the example function
        prompt_tokens = self.chat_prompt_tokens([dialog], return_tensor=False)


        prompt_length = len(prompt_tokens[0])

        prefix_tokens = self.tokenizer.encode('{"Reasoning": "', add_special_tokens=False) # TODO make this connected to the system message
        prompt_tokens[0] += prefix_tokens

        prompt_tokens = torch.tensor(prompt_tokens)
        prompt_tokens = prompt_tokens.to(self.device)

        outputs = self.model.generate(prompt_tokens, max_new_tokens=512)

        corrected_json_str = self.tokenizer.decode(outputs[0][prompt_length:])

        print(corrected_json_str)
        try:
            start_idx = corrected_json_str.find('{')
            end_idx = corrected_json_str.rfind('}')
            corrected_json_str = corrected_json_str[start_idx:end_idx+1]
            corrected_json_obj = json.loads(corrected_json_str)
            return corrected_json_obj
        except Exception as e:
            if verbose:
                print(f'Warning: could not parse corrected JSON from generated output. Error: {str(e)}')
            return None
