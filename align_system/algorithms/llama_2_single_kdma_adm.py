import json
import re
import random
import os
import pathlib
from align_system.algorithms.abstracts import AlignedDecisionMaker

from jinja2.exceptions import TemplateError

from rich.highlighter import JSONHighlighter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from align_system.utils import logging


from align_system.similarity_measures import build_force_choice_func


log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()

# TODO make this configurable from the config
kdmas = {
    'basic_knowledge',
    'fairness',
    'protocol_focus',
    'time_pressure',
    'risk_aversion',
    'utilitarianism',
    'mission',
    'denial',
    'moral_deservingness',
    'lives_saved',
    'continuation_of_care',
    'maximization'
}

kdma_remapping = {
    'basicknowledge': 'basic_knowledge',
    'protocolfocus': 'protocol_focus',
    'riskaversion': 'risk_aversion',
    'moraldeservingness': 'moral_deservingness',
    'continuationofcare': 'continuation_of_care',
    'livesaved': 'lives_saved',
    'timepressure': 'time_pressure',
}

# default_system_messages_path=os.path.join(
#     pathlib.Path(__file__).parent.absolute(), '..',
#     'prompt_engineering/bbn_alignment_system_messages_v1')

# NOTE temporary way to change which system messages are used
# TODO make this configurable from the config
default_system_messages_path=os.path.join(
    pathlib.Path(__file__).parent.absolute(), '..',
    'prompt_engineering/single_kdma_adm_system_messges')

chat_template_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(), '..',
    'prompt_engineering/chat_templates')

def load_system_message(alignment=None,
                        system_messages_path=default_system_messages_path):
    if alignment is None:
        file_name = 'baseline.txt'
    else:
        sorted_kdmas = sorted(alignment.keys())

        alignment_string = '-'.join(
            '{}-{}'.format(alignment[k], kdma_remapping.get(k, k))
            for k in sorted_kdmas)

        file_name = f'{alignment_string}.txt'

    with open(os.path.join(system_messages_path, file_name), 'r') as f:
        system_message = f.read()
    return system_message


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


class Llama2SingleKDMAADM(AlignedDecisionMaker):

    def __init__(self, device='cuda', hf_model='meta-llama/Llama-2-7b-chat-hf', precision='full', temperature=0.7, **kwargs):
        self.device = device
        self.hf_model = hf_model
        self.temperature = temperature
        self.chat_template = kwargs.get('chat_template', None)

        assert precision in ['full', 'half'], "precision must be either 'full' or 'half'."
        self.precision = torch.float32 if precision == 'full' else torch.float16

        self.model = None
        self.tokenizer = None


    def load_model(self, model=None, tokenizer=None):
        assert (model is None) == (tokenizer is None), "model and tokenizer must both be None or both be not None."
        if model is not None:
            print('Loading model and tokenizer from provided objects.')
            self.model = model
            self.tokenizer = tokenizer
        else:
            print('Loading model:', self.hf_model)
            if self.device == 'auto':
                self.model = AutoModelForCausalLM.from_pretrained(self.hf_model, torch_dtype=self.precision, device_map='auto')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.hf_model, torch_dtype=self.precision)
                self.model = self.model.to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)

            if self.chat_template is not None:
                with open(os.path.join(chat_template_path, self.chat_template), 'r') as f:
                    self.tokenizer.chat_template = f.read().replace('    ', '').replace('\n', '')



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


    def build_multiple_choice_dialog(self,
                                     question,
                                     options,
                                     system_message=None,
                                     json_format=STANDARD_MULTIPLE_CHOICE_JSON_FORMAT):
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

        return dialog

    def log_dialog(self, dialog):
        for e in dialog:
            if e.get('role') == 'system':
                color = 'yellow'
            else:
                color = 'blue'

            log.debug(f"[bold {color}]**{e.get('role')}**[/bold {color}]",
                      extra={"markup": True})
            log.debug(f"[{color}]{e.get('content')}[/{color}]",
                      extra={"markup": True, "highlighter": None})

    def respond_to_dialog(self, dialog, prefix=None):
        inference_pair = {}
        if prefix is None:
            prefix = '{"Reasoning": "'
        # prompt_tokens = self.chat_prompt_tokens([dialog], return_tensor=False)
        try:
            prompt_tokens = [self.tokenizer.apply_chat_template(dialog, tokenize=True)]
            inference_pair['input'] = self.tokenizer.apply_chat_template(dialog, tokenize=False)
        except TemplateError:
            new_dialog = []
            for message in dialog:
                if message['role'] == 'system':
                    message['role'] = 'user'

                if len(new_dialog) == 0:
                    new_dialog.append(message)
                    continue

                last_message = new_dialog[-1]
                if last_message['role'] == message['role']:
                    last_message['content'] += '\n\n' + message['content']
                else:
                    new_dialog.append(message)
            dialog = new_dialog
            print('INPUT\n', dialog)
            prompt_tokens = [self.tokenizer.apply_chat_template(dialog, tokenize=True)]
            inference_pair['input'] = self.tokenizer.apply_chat_template(dialog, tokenize=False)

        prompt_length = len(prompt_tokens[0])

        if prefix is not None:
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            prompt_tokens[0] += prefix_tokens

        prompt_tokens = torch.tensor(prompt_tokens)
        if self.device != 'auto':
            prompt_tokens = prompt_tokens.to(self.device)

        outputs = self.model.generate(prompt_tokens, return_dict_in_generate=True, output_scores=True, max_new_tokens=512, temperature=self.temperature, do_sample=True)

        # Print the generated model output
        generated_output = self.tokenizer.decode(outputs.sequences[0][prompt_length:])
        inference_pair['output'] = generated_output

        return generated_output, inference_pair

    def respond_to_dialogs_batched(self, dialogs, prefixes=None):
        # dialogs = [self.build_multiple_choice_dialog(*args) for args
        #            in zip(questions, option_lists, system_messages)]

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
            max_new_tokens=512,
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

    def aligned_decision_maker(self, question, choices, target_kdmas, n_positive_samples=5, n_negative_sampels=5, shuffle=True, baseline=False, n_retries=3):
        inference_pairs = []
        if not baseline:
            unsupported_kdmas = {kdma_remapping.get(k, k)
                                 for k in target_kdmas.keys()} - kdmas
            if len(unsupported_kdmas) > 0:
                raise RuntimeError(f"KDMA(s) {unsupported_kdmas} not supported.")

        prefix = '{"Reasoning": "Because'

        responses = []

        logged_aligned_dialog = False
        logged_inverse_misaligned_dialog = False
        for _ in range(n_positive_samples):
            if baseline:
                system_message = load_system_message()
                system_message_keys = 'baseline'

            else:
                system_message_keys = {kdma: 'high' if value > 5 else 'low'
                                    for kdma, value in target_kdmas.items()}
                system_message = load_system_message(system_message_keys)

            indecies = list(range(len(choices)))
            if shuffle:
                random.shuffle(indecies)
            shuffled_choices = [choices[i] for i in indecies]

            dialog = self.build_multiple_choice_dialog(
                question,
                shuffled_choices,
                system_message=system_message)

            if not logged_aligned_dialog:
                log.debug("[bold]*ALIGNED DIALOG*[/bold]",
                          extra={"markup": True})
                self.log_dialog(dialog)
                logged_aligned_dialog = True

            good_parse = False
            for i in range(n_retries):
                high_response, inference_pair = self.respond_to_dialog(dialog, prefix=prefix)
                inference_pairs.append({**inference_pair, **{'aligned': True, 'attempt': i}})
                try:
                    reasoning, answer_idx, parse_method = Llama2SingleKDMAADM.parse_generated_output(high_response, len(choices))
                    good_parse = True
                    break
                except RuntimeError as e:
                    pass

            if not good_parse:
                reasoning, answer_idx, parse_method = Llama2SingleKDMAADM.bert_similarity_parse(high_response, shuffled_choices)

            print('CHOSEN ANSWER IDX', answer_idx, shuffled_choices)
            assert answer_idx is not None, f'Failed to parse answer index from generated output: {low_response}'

            responses.append({
                'response': high_response,
                'reasoning': reasoning,
                'answer_idx': answer_idx,
                'shuffle_indecies': indecies,
                'alignment': system_message_keys,
                'aligned': True,
                'parse_method': parse_method,
            })

        for _ in range(n_negative_sampels):
            system_message_keys = {kdma: 'high' if not value > 5 else 'low'
                                    for kdma, value in target_kdmas.items()}

            indecies = list(range(len(choices)))
            if shuffle:
                random.shuffle(indecies)
            shuffled_choices = [choices[i] for i in indecies]

            inverse_misaligned_dialog = self.build_multiple_choice_dialog(
                question,
                shuffled_choices,
                system_message=load_system_message(system_message_keys))

            if not logged_inverse_misaligned_dialog:
                log.debug("[bold]*INVERSE MISALIGNED DIALOG*[/bold]",
                            extra={"markup": True})
                self.log_dialog(inverse_misaligned_dialog)
                logged_inverse_misaligned_dialog = True

            good_parse = False
            for i in range(n_retries):
                low_response, inference_pair = self.respond_to_dialog(inverse_misaligned_dialog, prefix=prefix)
                inference_pairs.append({**inference_pair, **{'aligned': True, 'attempt': i}})
                try:
                    reasoning, answer_idx, parse_method = Llama2SingleKDMAADM.parse_generated_output(low_response, len(choices))
                    good_parse = True
                    break
                except RuntimeError as e:
                    pass

            if not good_parse:
                reasoning, answer_idx, parse_method = Llama2SingleKDMAADM.bert_similarity_parse(low_response, shuffled_choices)

            assert answer_idx is not None, f'Failed to parse answer index from generated output: {low_response}'

            responses.append({
                'response': low_response,
                'reasoning': reasoning,
                'answer_idx': answer_idx,
                'shuffle_indecies': indecies,
                'alignment': system_message_keys,
                'aligned': False,
                'parse_method': parse_method,
            })

        return responses, inference_pairs


    @staticmethod
    def calculate_votes(responses, choices):
        # Initialize a list to keep track of votes for each choice, starting at 0
        choice_votes = [0] * len(choices)
        # Loop through each response
        for response in responses:
            answer_idx = response['answer_idx']

            # Continue to next response if answer_idx is None
            if answer_idx is None:
                continue

            # Try converting the answer_idx to an integer, skip to the next response if it fails
            try:
                answer_idx = int(answer_idx)
            except ValueError:
                continue

            # Skip this response if the answer_idx is out of bounds of the available choices
            if answer_idx >= len(choices):
                continue
            
            # TODO it may be beneficial to throw a warning or error if either of the above conditions are met

            # If there are shuffled indices provided, use them to find the original index of the answer
            if 'shuffle_indecies' in response:
                answer_idx = response['shuffle_indecies'][int(answer_idx)]

            # Get the alignment status of the response
            aligned = response['aligned']

            # If the response is aligned, increment the vote count for the chosen answer
            if aligned:
                choice_votes[answer_idx] += 1
            else:
                # If not aligned, distribute the vote fractionally among all other options
                # and subtract from the chosen option
                for i in range(len(choices)):
                    if i != answer_idx:
                        choice_votes[i] += 1/len(choices)
                    else:
                        choice_votes[i] -= 1/len(choices)
            # This ensures that each sampling has the same weight, but a negative vote affects all the choices that weren't selected

        # Normalize scores to ensure no negative totals and adjust based on the lowest score
        min_score = min(choice_votes) + 1e-6  # Add a small constant to avoid zero division
        choice_votes = [score - min_score for score in choice_votes]
        
        # Calculate the sum of all adjusted vote counts
        total = sum(choice_votes)
        # Normalize the vote counts to a percent distribution, rounded to six decimal points
        choice_votes = [round(score / total, 6) for score in choice_votes]

        # Return the final normalized, adjusted vote counts
        return choice_votes


    @staticmethod
    def parse_generated_output(generated_output, n_choices):
        parse_method = 'json'

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
                try:
                    answer_idx = int(str(parsed['Answer']))
                except ValueError:
                    pass
        except json.JSONDecodeError:
            pass



        if answer_idx is None:
            parse_method = 'string'
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

        if reasoning is None:
            reasoning = generated_output

        if answer_idx is None or answer_idx >= n_choices:
            raise RuntimeError(f'Failed to parse answer index < {n_choices} from generated output: {generated_output}')

        return reasoning, answer_idx, parse_method

    @staticmethod
    def bert_similarity_parse(generated_output, choices):
        print('BERT SIMILARITY PARSE')
        force_choice_func = build_force_choice_func('bert')
        answer_idx, _ = force_choice_func(generated_output, choices)
        print('ANSWER IDX', answer_idx, type(answer_idx))
        return generated_output, answer_idx, 'bert_similarity'

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

        log.debug(corrected_json_str, extra={"highlighter": JSON_HIGHLIGHTER})
        try:
            start_idx = corrected_json_str.find('{')
            end_idx = corrected_json_str.rfind('}')
            corrected_json_str = corrected_json_str[start_idx:end_idx+1]
            corrected_json_obj = json.loads(corrected_json_str)
            return corrected_json_obj
        except Exception as e:
            if verbose:
                log.warning(f'Warning: could not parse corrected JSON from generated output. Error: {str(e)}')
            return None

    def run_aligned_decision_maker_with_voting(
            self, prompt, choices, alignment_target, n_positive_samples=5, n_negative_samples=5, baseline=False, shuffle=False):
        responses, inference_pairs = self.aligned_decision_maker(
            prompt,
            choices,
            alignment_target,
            baseline=baseline,
            n_positive_samples=n_positive_samples,
            n_negative_sampels=n_negative_samples,
            shuffle=shuffle
        )

        try:
            choice_scores = Llama2SingleKDMAADM.calculate_votes(responses, choices)
        except Exception as e:
            log.warning(f"Error calculating votes: {e}")
            choice_scores = [None] * len(choices)

        log.debug("[bold]*RESPONSES*[bold]", extra={"markup": True})
        for i, ip in enumerate(inference_pairs):
            log.debug("[bold]*response {}*[bold]".format(i+1),
                      extra={"markup": True})
            log.debug(ip['output'])

        log.explain("[bold]*CHOICE SCORES*[/bold]",
                    extra={"markup": True})
        log.explain("\n".join([f"{c}: {s}" for c, s in zip(choices, choice_scores)]))

        results = {
            'prompt': prompt,
            'choice_scores': choice_scores,
            'responses': responses,
        }

        answer_idx = int(np.argmax(results['choice_scores']))
        reasoning = None

        for r in responses:
            assert r['answer_idx'] is not None
            assert int(r['answer_idx']) < len(r['shuffle_indecies'])

            if r['shuffle_indecies'][int(r['answer_idx'])] == answer_idx:
                reasoning = r['reasoning']
                break

        return reasoning, answer_idx, responses, inference_pairs

    def __call__(self, sample, target_kdma_values, **kwargs):
        prompt = sample['scenario']
        if sample['state'] is not None:
            prompt += f'\n{sample["state"]}'

        if 'retriever' in kwargs:
            # retriever_prompt = "How would you treat the following injuries: {}".format(prompt)
            retriever_prompt = "{}  {}".format(prompt, sample['probe'])

            retriever = kwargs['retriever']
            retrieved_nodes = retriever.retrieve(retriever_prompt)

            if 'summarizer' in kwargs:
                summarizer = kwargs['summarizer']
                summary = summarizer.synthesize(retriever_prompt, nodes=retrieved_nodes)

                log.explain("[bold] ** Retrieval Summary ** [/bold]",
                            extra={"markup": True})
                log.explain(summary)

                prompt += "\n#############\n{}\n#############".format(summary)

            else:
                prompt += "\n#############\n{}\n#############".format(
                    "\n#############\n".join((n.text for n in retrieved_nodes)))

            prompt += f'\nGiven the scenario and documentation above.. {sample["probe"]}'
        else:
            prompt += f'\n{sample["probe"]}'

        choices = sample['choices']

        labels = kwargs.get('labels', {})

        alignment_target = None
        if target_kdma_values is not None and len(target_kdma_values) > 0:
            target_kdma = next(iter(next(iter(filter(lambda x: len(x) > 0, labels))))) # get the frist key of the first label that is not empty

            for label in labels:
                assert len(label) == 0 or (target_kdma in label and len(label) == 1), f'All labels must have the same KDMA: labels={labels}'

            alignment_target = {
                target_kdma: target_kdma_values[target_kdma]
            }

        reasoning, answer_idx, responses, inference_pairs = self.run_aligned_decision_maker_with_voting(
            prompt,
            choices,
            alignment_target,
            n_positive_samples=kwargs.get('n_positive_samples', 5),
            n_negative_samples=kwargs.get('n_negative_samples', 5),
            baseline=kwargs.get('baseline', False),
            shuffle=kwargs.get('shuffle', False)
        )

        raw_data = {
            'params': {
                'model': self.hf_model,
                'temperature': self.temperature,
                'n_positive_samples': kwargs.get('n_positive_samples', 5),
                'n_negative_samples': kwargs.get('n_negative_samples', 5),
                'baseline': kwargs.get('baseline', False),
                'shuffle': kwargs.get('shuffle', False),
            },
            'inference_pairs': inference_pairs
        }

        return {
            'choice': int(answer_idx),
            'info': {
                'reasoning': reasoning,
                'responses': responses,
                'raw_data': raw_data,
            }
        }

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        from swagger_client.models import ActionTypeEnum

        kdma_name_map = {
            'MoralDesert': 'moral_deservingness',
            'maximization': 'maximization',
        }

        if alignment_target is None or len(alignment_target.kdma_values) == 0:
            target_kdma_values = {}
        else:
            alignment_target_dict = alignment_target.to_dict()
            target_kdma_values = {
                kdma_name_map[k['kdma']]: k['value'] * 10
                for k in alignment_target_dict.get('kdma_values', ())
            }

        scenario = '\nCHARACTERS:\n'

        for character in scenario_state.characters:
            scenario += f'{character.name}: {character.unstructured}\n'
            scenario += f'{character.name}\'s intent: {character.intent}\n\n'

        scenario += f'\nSITUATION:\n{scenario_state.unstructured}'

        state = None

        probe = ''

        choices = [
            action.unstructured
            for action in available_actions
        ]

        response = self.__call__({
            'scenario': scenario,
            'state': state,
            'probe': probe,
            'choices': choices},
            target_kdma_values,
            labels=[target_kdma_values]*len(choices),
            **kwargs)

        action_to_take = available_actions[response['choice']]

        if action_to_take.action_type == ActionTypeEnum.APPLY_TREATMENT:
            # If the additional required fields are already populated
            # for the action, don't need ask the LLM again
            if (action_to_take.parameters is None
                or not {'treatment', 'location'}.issubset(
                    action_to_take.parameters.keys())):
                action_to_take = self.populate_treatment_parameters(
                        scenario_state, action_to_take, alignment_target, **kwargs)
        elif action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            # If the additional required fields are already populated
            # for the action, don't need ask the LLM again
            if (action_to_take.character_id is None
                or action_to_take.parameters is None
                or not {'category'}.issubset(
                    action_to_take.parameters.keys())):
                action_to_take = self.populate_tagging_parameters(
                    scenario_state, action_to_take, alignment_target, **kwargs)
        elif action_to_take.action_type in {ActionTypeEnum.CHECK_ALL_VITALS,
                                            ActionTypeEnum.CHECK_PULSE,
                                            ActionTypeEnum.CHECK_RESPIRATION,
                                            ActionTypeEnum.MOVE_TO_EVAC}:
            # These actions require a `character_id`
            if action_to_take.character_id is None:
                action_to_take = self.generic_populate_character_id(
                    scenario_state, action_to_take, alignment_target, **kwargs)

        return action_to_take

    def populate_treatment_parameters(self, scenario_state, treatment_action, alignment_target, **kwargs):
        from align_system.prompt_engineering.common import (
            prepare_treatment_selection_prompt)
        from swagger_client.models import ActionTypeEnum, InjuryLocationEnum
        from align_system.utils import get_swagger_class_enum_values

        assert treatment_action.action_type == ActionTypeEnum.APPLY_TREATMENT

        character_id = treatment_action.character_id
        if character_id is None:
            # Need to populate character_id on treatment action
            treatment_action = self.generic_populate_character_id(
                scenario_state, treatment_action, alignment_target, **kwargs)

            character_id = treatment_action.character_id

        matching_characters = [c for c in scenario_state.characters
                               if c.id == character_id]

        assert len(matching_characters) == 1

        character_to_treat = matching_characters[0]

        available_supplies = [s for s in scenario_state.supplies if s.quantity > 0]

        treatment_prompt = prepare_treatment_selection_prompt(
            character_to_treat.unstructured,
            character_to_treat.vitals.to_dict(),
            [s.to_dict() for s in available_supplies])

        for _ in range(kwargs.get('answer_attempts', 5)):
            treatment_dialog =\
                self.build_multiple_choice_dialog(
                    treatment_prompt,
                    [s.to_dict() for s in available_supplies],
                    json_format=TREATMENT_MULTIPLE_CHOICE_JSON_FORMAT)

            log.debug("[bold]*TREATMENT DIALOG*[/bold]",
                      extra={"markup": True})
            self.log_dialog(treatment_dialog)

            raw_treatment_response, _ = self.respond_to_dialog(
                treatment_dialog)

            log.info("** ADM raw treatment response: {}".format(
                raw_treatment_response))

            parsed_treatment_output = self.attempt_generic_parse(  # noqa
                raw_treatment_response, ['Reasoning', 'Answer', 'Location'])  # noqa

            if parsed_treatment_output is not None:
                treatment_idx = parsed_treatment_output['Answer']

                if len(available_supplies) <= treatment_idx:
                    log.info('** Selected treatment_idx out of range of '
                             'available treatment options, retrying!')
                    continue

                treatment = available_supplies[treatment_idx].type  # noqa

                treatment_location = parsed_treatment_output['Location']

                if treatment_action.parameters is None:
                    treatment_action.parameters = {}

                treatment_action.parameters['treatment'] = treatment

                valid_treatment_locations = get_swagger_class_enum_values(
                    InjuryLocationEnum)

                if not isinstance(treatment_location, str):
                    # If type is int, could be an index into the
                    # action_to_take)locations provided in the system
                    # action_to_take)prompt, consider handling in the
                    # action_to_take)future
                    log.warning("*** Treatment location value is not a string"
                                ", retrying!")
                    continue
                if treatment_location in valid_treatment_locations:
                    treatment_action.parameters['location'] = treatment_location
                else:
                    # Ensure that the treatment location is valid
                    _, treatment_loc_idx, _ = self.bert_similarity_parse(
                        treatment_location, valid_treatment_locations)

                    treatment_action.parameters['location'] =\
                        valid_treatment_locations[treatment_loc_idx]

                break
            else:
                log.info('** Failed to parse treatment')

        return treatment_action

    def populate_tagging_parameters(self, scenario_state, tagging_action, alignment_target, **kwargs):
        from align_system.prompt_engineering.common import (
            prepare_tagging_selection_prompt)
        from swagger_client.models import ActionTypeEnum, CharacterTagEnum
        from align_system.utils import get_swagger_class_enum_values

        assert tagging_action.action_type == ActionTypeEnum.TAG_CHARACTER
        # Ask the system to specify which triage tag to apply

        untagged_characters = [c for c in scenario_state.characters if c.tag is None]

        tagging_prompt = prepare_tagging_selection_prompt(
            [c.to_dict() for c in untagged_characters],
            get_swagger_class_enum_values(CharacterTagEnum))

        for _ in range(kwargs.get('answer_attempts', 5)):
            tagging_dialog = self.build_multiple_choice_dialog(
                tagging_prompt,
                [c.unstructured.strip()
                 for c in untagged_characters],
                json_format=TAGGING_MULTIPLE_CHOICE_JSON_FORMAT)

            log.debug("[bold]*TAGGING DIALOG*[/bold]",
                      extra={"markup": True})
            self.log_dialog(tagging_dialog)

            raw_tagging_response, _ = self.respond_to_dialog(
                tagging_dialog)

            log.info("** ADM raw tagging response: {}".format(
                raw_tagging_response))

            parsed_tagging_output = self.attempt_generic_parse(  # noqa
                raw_tagging_response, ['Reasoning', 'Answer', 'Tag'])  # noqa

            if parsed_tagging_output is not None:
                if len(untagged_characters) == 1:
                    log.debug("** Force selecting only available character")
                    character_idx = 0
                else:
                    character_idx = parsed_tagging_output['Answer']

                    if not isinstance(character_idx, int):
                        log.warning('** character_idx ({}) not an integer'
                                    ', retrying!'.format(character_idx))
                        continue

                    if len(untagged_characters) <= character_idx:
                        log.info('** Selected character_idx out of range of '
                                 'available treatment options, retrying!')
                        continue

                character_to_tag_id = untagged_characters[character_idx].id  # noqa

                tag = parsed_tagging_output['Tag']
                if not isinstance(tag, str):
                    log.warning("** Selected tag ({}) not of type string"
                                ", retrying!".format(tag))
                    continue

                # Populate required parameters for tagging action
                tagging_action.character_id = character_to_tag_id

                if tagging_action.parameters is None:
                    tagging_action.parameters = {}

                tagging_action.parameters['category'] = tag

                break
            else:
                log.info('** Failed to parse tagging')

        return tagging_action

    def generic_populate_character_id(self, scenario_state, initial_action, alignment_target, **kwargs):
        from swagger_client.models import ActionTypeEnum
        from align_system.prompt_engineering.common import (
            prepare_character_selection_prompt)
        character_selection_prompt = prepare_character_selection_prompt(
            initial_action)

        filtered_characters = []
        for c in scenario_state.characters:
            if initial_action.action_type in {ActionTypeEnum.CHECK_ALL_VITALS,
                                              ActionTypeEnum.CHECK_PULSE,
                                              ActionTypeEnum.CHECK_RESPIRATION}:
                # Don't allow the ADM to check vitals on
                # a character that's already been "visited"
                if c.visited:
                    continue

            filtered_characters.append(c)

        for _ in range(kwargs.get('answer_attempts', 5)):
            character_selection_dialog = self.build_multiple_choice_dialog(
                character_selection_prompt,
                [c.unstructured.strip()
                 for c in filtered_characters])

            log.debug("[bold]*CHARACTER SELECTION DIALOG*[/bold]",
                      extra={"markup": True})
            self.log_dialog(character_selection_dialog)

            raw_character_selection_response, _ = self.respond_to_dialog(
                character_selection_dialog)

            log.info("** ADM raw character_selection response: {}".format(
                raw_character_selection_response))

            parsed_character_selection_output = self.attempt_generic_parse(  # noqa
                raw_character_selection_response, ['Reasoning', 'Answer'])  # noqa

            if parsed_character_selection_output is not None:
                if len(filtered_characters) == 1:
                    log.debug("** Force selecting only available character")
                    character_idx = 0
                else:
                    character_idx = parsed_character_selection_output['Answer']

                    if not isinstance(character_idx, int):
                        log.warning('** character_idx ({}) not an integer'
                                    ', retrying!'.format(character_idx))
                        continue

                    if len(filtered_characters) <= character_idx:
                        log.warning('** Selected character_idx out of range of '
                                    'available treatment options, retrying!')
                        continue

                character_id = filtered_characters[character_idx].id  # noqa

                # Populate required parameters for character_selection action
                initial_action.character_id = character_id

                break
            else:
                log.info('** Failed to parse character selection')

        return initial_action
