import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import random


kdmas = [
    'basic_knowledge',
    'fairness',
    'protocol_focus',
    'risk_aversion',
    'utilitarianism',
]


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
                system_dialog = {
                    "role": dialog[1]["role"],
                    "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"]
                }
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


    def answer_multiple_choice(self, question, options, system_message=None, prefix=None, json_format=STANDARD_MULTIPLE_CHOICE_JSON_FORMAT):
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


    def respond_to_dialog_batched(self, dialogs, prefixes=None, max_new_tokens=512):
        prompt_token_lists = [
            self.chat_prompt_tokens([dialog], return_tensor=False)
            for dialog in dialogs
        ]


        prompt_lengths = [
            len(prompt_tokens[0])
            for prompt_tokens in prompt_token_lists
        ]

        if prefixes is not None:
            for prompt_tokens, prefix in zip(prompt_token_lists, prefixes):
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                prompt_tokens[0] += prefix_tokens


        prompt_token_lists = [
            torch.tensor(prompt_tokens).to(self.device)
            for prompt_tokens in prompt_token_lists
        ]
        
        max_length = max([prompt_tokens.size(1) for prompt_tokens in prompt_token_lists])

        pad_token_id = self.tokenizer.pad_token_id
        # Pad each sequence to the max length
        padded_prompt_token_lists = [
            torch.nn.functional.pad(prompt_tokens, (max_length - prompt_tokens.size(1), 0), value=pad_token_id)
            for prompt_tokens in prompt_token_lists
        ]

        # Stack the padded sequences
        stacked_prompt_tokens = torch.cat(padded_prompt_token_lists, dim=0)

        # Generate outputs for all dialogs in a batch
        outputs = self.model.generate(
            stacked_prompt_tokens, 
            return_dict_in_generate=True, 
            output_scores=True, 
            max_new_tokens=max_new_tokens,
            temperature=self.temperature
        )

        # Split the sequences based on prompt lengths 
        split_outputs = torch.split(outputs.sequences, 1, dim=0)

        # Decode each output based on its corresponding prompt length
        generated_outputs = [
            self.tokenizer.decode(output[0][max(prompt_lengths):])
            for output in split_outputs
        ]

        # split on </s> and remove trailing characters
        generated_outputs = [
            generated_output.split('</s>')[0].strip()
            for generated_output in generated_outputs
        ]

        return generated_outputs

    
    # answer_multiple_choice(self, question, options, system_message=None, prefix=None, json_format=STANDARD_MULTIPLE_CHOICE_JSON_FORMAT):
        
    def answer_multiple_choice(self, question, options, system_message, prefix=None, max_new_tokens=512):
        batched_question = type(question) is list
        batched_options = type(options[0]) is list
        batched_system_message = type(system_message) is list
        batched_prefix = type(prefix) is list or prefix is None
        
        assert batched_question == batched_options == batched_system_message == batched_prefix, "All inputs must be batched or not batched."
        
        all_batched = batched_question and batched_options and batched_system_message and batched_prefix
        
        if not all_batched:
            questions = [question]
            option_lists = [options]
            system_messages = [system_message]
            prefixes = [prefix] if prefix is not None else None
        else:
            questions = question
            option_lists = options
            system_messages = system_message
            prefixes = prefix

        formatted_option_lists = [[f'({i}) {option}' for i, option in enumerate(options)] for options in option_lists]

        contents = [f'{question} {formatted_options}' for question, formatted_options in zip(questions, formatted_option_lists)]

        dialogs = [
            [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
            for system_message, content in zip(system_messages, contents)
        ]

        generated_outputs = self.respond_to_dialog_batched(dialogs, prefixes=prefixes, max_new_tokens=max_new_tokens)
        
        if not all_batched:
            return generated_outputs[0]
        return generated_outputs



    def aligned_decision_maker(self, question, choices, target_kdmas, system_message_provider, prefix='{"Reasoning": "Because', n_samples=5, inverse_misaligned=True, shuffle=True, baseline=False, batch_size=5, max_new_tokens=512, include_prompt=False):
        assert len(target_kdmas) == 1, "Only one KDMA can be targeted at a time, but received: {}".format(target_kdmas)
        
        kdma = list(target_kdmas.keys())[0]
        
        assert kdma in kdmas, f"KDMA {kdma} not supported."
        
        results = []
        
        inputs = []

        for _ in range(n_samples):
            system_message_keys = [kdma, 'high' if target_kdmas[kdma] > 5 else 'low']
            
            indecies = list(range(len(choices)))
            if shuffle:
                random.shuffle(indecies)
            shuffled_choices = [choices[i] for i in indecies]

            system_message = system_message_provider(system_message_keys[0], system_message_keys[1])
            
            if baseline:
                system_message = system_message_provider('baseline', None)
                system_message_keys[1] = 'baseline'
            
            assert not (baseline and inverse_misaligned), "Baseline and inverse misaligned cannot be used together."
            
            def callback(high_response, 
                    kdma=kdma,
                    system_message=system_message,
                    alignment=system_message_keys[1],
                    shuffled_choices=shuffled_choices,
                    indecies=indecies,
                ):
                reasoning, answer_idx = LLMChatBaseline.parse_generated_output(high_response)
                result = {
                    'response': high_response,
                    'reasoning': reasoning,
                    'answer_idx': answer_idx,
                    'shuffle_indecies': indecies,
                    'kdma': kdma,
                    'alignment': alignment,
                    'aligned': True,
                }
                if include_prompt:
                    result['prompt'] = {
                        'system_message': system_message,
                        'question': question,
                        'shuffled_choices': shuffled_choices,
                        'prefix': prefix,
                    }

                results.append(result)
                
            inputs.append({
                'question': question,
                'shuffled_choices': shuffled_choices,
                'system_message': system_message,
                'prefix': prefix,
                'callback': callback,
            })
            
            if inverse_misaligned:
                system_message_keys = [kdma, 'high' if not target_kdmas[kdma] > 5 else 'low']

                indecies = list(range(len(choices)))
                if shuffle:
                    random.shuffle(indecies)
                shuffled_choices = [choices[i] for i in indecies]

                def callback(
                    low_response,
                    kdma=kdma,
                    system_message=system_message,
                    alignment=system_message_keys[1],
                    shuffled_choices=shuffled_choices,
                    indecies=indecies,
                ):
                    reasoning, answer_idx = LLMChatBaseline.parse_generated_output(low_response)
                    result = {
                        'response': low_response,
                        'reasoning': reasoning,
                        'answer_idx': answer_idx,
                        'shuffle_indecies': indecies,
                        'kdma': kdma,
                        'alignment': alignment,
                        'aligned': False,
                    }
                    if include_prompt:
                        result['prompt'] = {
                            'system_message': system_message,
                            'question': question,
                            'shuffled_choices': shuffled_choices,
                            'prefix': prefix,
                        }
                    results.append(result)
                
                inputs.append({
                    'question': question,
                    'shuffled_choices': shuffled_choices,
                    'system_message': system_message_provider(system_message_keys[0], system_message_keys[1]),
                    'prefix': prefix,
                    'callback': callback,
                })
            
        for i in range(0, len(inputs), batch_size):
            responses = self.answer_multiple_choice(
                questions=[sample['question'] for sample in inputs[i:i+batch_size]],
                option_lists=[sample['shuffled_choices'] for sample in inputs[i:i+batch_size]],
                system_messages=[sample['system_message'] for sample in inputs[i:i+batch_size]],
                prefixes = [sample['prefix'] for sample in inputs[i:i+batch_size]],
                max_new_tokens=max_new_tokens
            )
            
            callbacks = [sample['callback'] for sample in inputs[i:i+batch_size]]
            
            for response, callback in zip(responses, callbacks):
                callback(response)
        
        return results

    @staticmethod
    def calculate_votes(responses, choices, flip_alignment=False):
        choice_votes = [0] * len(choices)
        for response in responses:
            answer_idx = response['answer_idx']
            if answer_idx is None or answer_idx > len(choices):
                continue
            
            if 'shuffle_indecies' in response:
                answer_idx = response['shuffle_indecies'][answer_idx]
            
            aligned = response['aligned']
            
            if flip_alignment:
                aligned = not aligned
            
            if aligned: 
                choice_votes[answer_idx] += 1
            else:
                for i in range(len(choices)):
                    if i != answer_idx:
                        choice_votes[i] += 1/len(choices)
                    else:
                        choice_votes[i] -= 1/len(choices)
        
        min_score = min(choice_votes) + 1e-6
        choice_votes = [score - min_score for score in choice_votes]
        total = sum(choice_votes)
        choice_votes = [round(score / total, 6) for score in choice_votes]
        
        return choice_votes
    

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
                if field == 'Answer':
                    if m := re.search(r'\d+', parsed_field):  # noqa
                        parsed_field = m.group(0)

                    try:
                        parsed_field = int(parsed_field)
                    except ValueError:
                        # Failed to parse
                        return None

            parsed_output[field] = parsed_field

        return parsed_output
