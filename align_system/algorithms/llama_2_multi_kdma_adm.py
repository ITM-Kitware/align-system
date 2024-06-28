import json
import re
import random
import os
import yaml
import pathlib
from align_system.algorithms.abstracts import AlignedDecisionMaker

from align_system.algorithms.lib.chat.chat_language_model import ChatLanguageModel
from typing import Union, List, Dict, Tuple, Optional, TextIO

from jinja2.exceptions import TemplateError

from rich.highlighter import JSONHighlighter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from align_system.utils import logging


from align_system.similarity_measures import build_force_choice_func
import IPython


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


# NOTE temporary way to change which system messages are used
# TODO make this configurable from the config
default_system_messages_path=os.path.join(
    pathlib.Path(__file__).parent.absolute(), '..',
    'prompt_engineering/multi_kdma_adm_system_messges')

chat_template_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(), '..',
    'prompt_engineering/chat_templates')

def read_template(template_file_name: str, template_dir=default_system_messages_path) -> str:
    with open(os.path.join(template_dir, template_file_name), 'r') as f:
        template = f.read()
    return template

STANDARD_MULTIPLE_CHOICE_JSON_FORMAT = "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>}\\n"

TREATMENT_MULTIPLE_CHOICE_JSON_FORMAT = "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>, \"Location\": \"<Specific location on the body where the treatment should be applied, one of: 'right forearm', 'left forearm', 'right calf', 'left calf', 'right thigh', 'left thigh', 'right stomach', 'left stomach', 'right bicep', 'left bicep', 'right shoulder', 'left shoulder', 'right side', 'left side', 'right chest', 'left chest', 'right wrist', 'left wrist', 'left face', 'right face', 'left neck', 'right neck', 'unspecified'>\"}\\n"

TAGGING_MULTIPLE_CHOICE_JSON_FORMAT = "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>, \"Tag\": \"<Specific medical triage tag to apply, one of: 'MINIMAL', 'DELAYED', 'IMMEDIATE', 'EXPECTANT'>\"}\\n"


class Llama2MultiKDMAADM(AlignedDecisionMaker, ChatLanguageModel):

    def __init__(self, device='cuda', hf_model='meta-llama/Llama-2-7b-chat-hf', precision='full', temperature=0.7, do_sample=True, **kwargs):
        self.device = device
        self.hf_model = hf_model
        self.temperature = temperature
        self.do_sample = do_sample
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

        outputs = self.model.generate(
            prompt_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
            temperature=self.temperature,
            do_sample=self.do_sample)

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
            temperature=self.temperature,
            do_sample=self.do_sample)

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

    def predict_outcomes(self, scenario, probe, choices, **kwargs):
        """
        Predicts outcomes for given scenario, probe and choices.

        :param scenario: Scenario text.
        :param probe: Probe text.
        :param choices: Choices text.
        :param kwargs
        :return: List of generated predictions.
        """
        template = 'pred_outcome.txt'
        templates = [read_template(template) for choice in choices] # TODO - make this work with a single string
        substitution_dicts = [
                {
                    'scenario': scenario,
                    'probe': probe, 
                    'choice': choice,
                }
                for choice in choices 
            ]
        return self.generate_from_template(
            templates,
            substitution_dicts,
            log_file=None, # Add new logging after call
            max_tokens=kwargs.get('max_new_tokens', 512),
            temperature=self.temperature
        )


    def predict_kdma_values(self, scenario_text, probe_text, choice_texts,
                            predicted_outcomes, **kwargs):
        """
        Predicts KDMA scores each choice text under the given scenario and probe.

        :param scenario_text: Scenario text.
        :param probe_text: Probe text.
        :param choice_texts: Choices text.
        :param predicted_outcomes: Predicted outcomes.
        :param kwargs
        :return: KDMA predictions. If generate_reasoning is True, return predictions and reasonings.
        """

        # TODO - properly add these to kwargs
        generate_reasoning=kwargs.get('generate_reasoning', True)
        log_file = None 
        max_new_tokens = kwargs.get('max_new_tokens', 512) 
        temperature = self.temperature
        template = 'pred_kdma_RO.txt' 
        kdma_descriptions_file = 'test_kdma_descriptions.yml'

        choice_ids = [f'choice_{i}' for i in range(len(choice_texts))]
        substitutions = []
        info = []
        
        kdma_descriptions_file_path = os.path.join(default_system_messages_path, kdma_descriptions_file)
        with open(kdma_descriptions_file_path, 'r') as f:
            kdma_descriptions = yaml.load(f, Loader=yaml.FullLoader)
        
        if predicted_outcomes is None:
            predicted_outcomes = [None] * len(choice_texts)
        
        for choice_id, choice, outcome in zip(choice_ids, choice_texts, predicted_outcomes):
            for kdma, kdma_info in kdma_descriptions.items():
                substitution = {
                    'kdma': kdma_info['name'],
                    'kdma_description': kdma_info['description'],
                    'scenario': scenario_text,
                    'probe': probe_text,
                    'choice': choice,
                }
                
                if outcome is not None:
                    substitution['outcome'] = outcome
                    
                substitutions.append(substitution)
                info.append((choice_id, kdma))
        
        def parse_kdma_score_response(response: str) -> Dict[str, Union[float, str]]:
            """
            Parses KDMA score response.

            :param response: Response to parse.
            :return: Dictionary with KDMA score and reasoning if generate_reasoning.
            """
            if generate_reasoning:
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                response_json = json.loads(response[start_idx:end_idx+1])
                assert 'score' in response_json, 'score not found in response'
                assert 'reasoning' in response_json, 'reasoning not found in response'
            else:
                # find the first numeric character
                char = None
                for c in response:
                    if c.isnumeric():
                        char = c
                        break
                assert char is not None, 'Could not find numeric character in response'
                response_json = {
                    'score': float(response[response.find(char):])
                }                
            return response_json
        
        templates = [read_template(template) for sub in substitutions] # TODO - make this work with a single string
        generations = self.generate_from_template(
            templates,
            substitutions,
            parse_kdma_score_response,
            log_file=log_file,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        predicted_kdma_values = {}
        reasonings = {}
        for (choice_id, kdma), generation in zip(info, generations):
            predicted_choice_kdmas = predicted_kdma_values.get(choice_id, {})
            predicted_kdma_values[choice_id] = predicted_choice_kdmas
            
            choice_reasonings = reasonings.get(choice_id, {})
            reasonings[choice_id] = choice_reasonings
            
            predicted_choice_kdmas[kdma] = generation['score']
            
            if generate_reasoning:
                choice_reasonings[kdma] = generation['reasoning']
        
        predicted_kdma_values = [
            predicted_kdma_values[choice_id]
            for choice_id in choice_ids
        ]
        if generate_reasoning:
            reasonings = [
                reasonings[choice_id]
                for choice_id in choice_ids
            ]
        
        if generate_reasoning:
            return predicted_kdma_values, reasonings
        else:
            return predicted_kdma_values


    def run_multi_aligned_decision_maker(
            self, probe, prompt, choices, target_kdma_values, **kwargs):
        n_samples=kwargs.get('n_samples', 5)

        predicted_kdma_values_samples = []
        generated_reasoning_samples = []

        for _ in range(n_samples):

            # Predict outcomes of each choice is specified
            predicted_outcomes = None
            if 'predict_outcomes'in kwargs:
                predicted_outcomes = self.predict_outcomes(
                    prompt, 
                    probe, 
                    choices, 
                    **kwargs)
                # TODO - Log predicted outcomes
            else:
                predicted_outcomes = None

            # Predict KDMA values
            predicted_kdma_values, generated_reasoning = self.predict_kdma_values(
                prompt, 
                probe, 
                choices,
                predicted_outcomes,
                **kwargs
            )
            # TODO log

            # add to samples
            if not predicted_kdma_values_samples:
                predicted_kdma_values_samples = [ \
                    {kdma: [predicted_kdma_values[choice_idx][kdma]] for kdma in predicted_kdma_values[choice_idx].keys()} \
                    for choice_idx in range(len(choices))]
            else:
                for choice_idx in range(len(choices)):
                    for kdma in predicted_kdma_values[choice_idx].keys():
                        predicted_kdma_values_samples[choice_idx][kdma].append(predicted_kdma_values[choice_idx][kdma])
            generated_reasoning_samples.append(generated_reasoning)

        # mean reduction over samples
        predicted_kdma_values = [ \
                {kdma: sum(sample[kdma]) / len(sample[kdma]) for kdma in sample.keys()} \
                for sample in predicted_kdma_values_samples]

        def mse(target_kdma_values, predicted_kdma_values):
            kdmas = set(target_kdma_values.keys()) & set(predicted_kdma_values.keys())
            
            if len(kdmas) == 0:
                return 0
        
            return sum([(target_kdma_values[kdma] - predicted_kdma_values[kdma])**2 for kdma in kdmas]) / len(kdmas)

        # find index of min mse
        choice_idx = 0
        min_mse = float('inf')
        for i, choice in enumerate(choices):
            mse_ = mse(target_kdma_values, predicted_kdma_values[i])
            if mse_ < min_mse:
                min_mse = mse_
                choice_idx = i

        print('CHOSEN ANSWER IDX', choice_idx, choices)

        return {
            'choice': choice_idx,
            'info': {
                'reasoning': 'placeholder',  # TODO
                'predicted_outcomes': predicted_outcomes,
                'predicted_kdmas': predicted_kdma_values,
                'generated_reasoning': generated_reasoning
                }
            }

    def __call__(self, sample, target_kdma_values, **kwargs):
        prompt = sample['scenario']
        if sample['state'] is not None:
            prompt += f'\n{sample["state"]}'

        probe = sample['probe']
        choices = sample['choices']

        labels = kwargs.get('labels', {})

        alignment_target = None
        if target_kdma_values is not None and len(target_kdma_values) > 0:
            target_kdmas = list(next(iter(filter(lambda x: len(x) > 0, labels))).keys()) # get all keys of the first label that is not empty

            for label in labels:
                assert len(label) == 0 or (all(target_kdma in label for target_kdma in target_kdmas) and len(label) == len(target_kdmas)), \
                    f'All labels must have the same KDMA: labels={labels}'

            alignment_target = {}
            for target_kdma in target_kdmas:
                alignment_target[target_kdma] = target_kdma_values[target_kdma]


        decision = self.run_multi_aligned_decision_maker(
            prompt,
            probe,
            choices,
            alignment_target,
            **kwargs
        )

        raw_data = {
            'params': {
                'model': self.hf_model,
                'temperature': self.temperature,
            }
        }

        decision['info']['raw_data'] = raw_data

        return(decision)


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
        action_to_take.justification = response['info'].get('reasoning')

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

        if isinstance(character_to_treat.vitals, dict):
            vitals_dict = character_to_treat.vitals
        else:
            vitals_dict = character_to_treat.vitals.to_dict()

        treatment_prompt = prepare_treatment_selection_prompt(
            character_to_treat.unstructured,
            vitals_dict,
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
                try:
                    treatment_idx = int(parsed_treatment_output['Answer'])
                except ValueError:
                    log.warning('** Treatment index not an integer, retrying!')
                    continue

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

                valid_tags = get_swagger_class_enum_values(CharacterTagEnum)
                if tag not in valid_tags:
                    log.warning("** Selected tag ({}) is not a valid tag"
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
