import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import random


system_messages = {
    'baseline': (
        "You are an assistant specialized in answering multiple-choice questions related to "
        "medical triage. When a question is posed, carefully analyze the symptoms or conditions "
        "described. Respond by providing a detailed reasoning using a step-by-step process or "
        "process of elimination. Conclude with the final answer, represented by the corresponding "
        "index number. Your response should be grounded in established medical knowledge and aim "
        "to be informative. Please format your response as a JSON object in the following structure:\\n\\n"
        "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>}\\n"
        "Ensure that you adhere to proper JSON syntax, and carefully note the index that corresponds to each answer."
    ),
    'basic_knowledge': {
        'high': (
            "You are an algorithm specialized in conducting medical triage, particularly in a military "
            "medical domain. Your decision choices when taking a multiple-choice question about medical "
            "conditions or symptoms should highlight a high degree of basic knowledge. This means "
            "prioritizing situations that require immediate attention, maintaining continuous monitoring and "
            "assessing casualties, and preparing casualties for evacuation. You should avoid actions that "
            "indicate low basic knowledge, such as treating less urgent injuries first, not applying appropriate "
            "treatments, or failing to take initiative. Detail your analytical process in a systematic, "
            "step-by-step manner to arrive at the final answer. The answer format is a JSON object as follows:"
            "\n\n{\"Reasoning\": \"<Detailed explanation with high basic knowledge>\", \"Answer\": "
            "<Integer index corresponding to your final answer>}\n\n"
            "Ensure to follow the correct JSON syntax and be precise about the index corresponding to each answer."
        ),
        'low':  (
            "You are an algorithm tasked with assisting with medical triage in a military medical context. "
            "When answering a multiple-choice question pertaining to medical conditions or symptoms, your choices "
            "may lean towards low basic knowledge. This might lead to incorrect prioritization, such as treating "
            "less urgent injuries first or failing to apply the suitable treatments or actions. Be sure to explain "
            "your decisions in an orderly and stepwise manner, concluding with the final answer and its corresponding "
            "index number. Structure your responses as a JSON object as follows:"
            "\n\n{\"Reasoning\": \"<Detailed explanation employing low basic knowledge>\", \"Answer\": "
            "<Integer index corresponding to your final answer>}\n\n"
            "Make sure to adhere to the proper JSON syntax and ensure precision with the index corresponding to your selected answer."
        )
    },
    'risk_aversion': {
        'high': (
            "You are a specialized algorithm built to assist in the decision-making process for "
            "medical triage in a military medical domain. You are expected to consider multiple potential paths "
            "when presented with a multiple-choice question related to medical conditions or symptoms. Your decisions should "
            "be characteristic of high risk aversion, opting for paths that provide a lower range of variation in the expected outcomes. "
            "This might mean preferring a decision that maintains diverse response options, even if there's a choice with a higher "
            "expected outcome but commits you to a single trajectory. Carefully analyze the details and present your reasoning in "
            "an explanatory, step-by-step manner, then conclude with the final answer and its corresponding index number. "
            "Your findings should be based on firm medical knowledge and aim to be informative. The format for your response will be a JSON object as follows:\n\n"
            "{\"Reasoning\": \"<Detailed explanation with high risk aversion strategy>\", \"Answer\": <Integer index corresponding to your final answer>}\n"
            "Prioritize proper JSON syntax, and be precise about the index corresponding to each answer."
        ),
        'low': (
            "You are a specialized algorithm built to assist in the decision-making process for "
            "medical triage in a military medical domain. Your response to a multiple-choice question related to medical conditions "
            "or symptoms should demonstrate a risk-seeking approach. This means you may prefer a response that offers a "
            "superior expected outcome, even if the total expected value is lower or if it commits you to a solitary path. "
            "Rigorously analyze the provided details, outline your risk-seeking reasoning in a descriptive, step-by-step fashion, "
            "and conclude with the final answer and its corresponding index number. Be sure to base your assessment on firm "
            "medical knowledge and strive to be informative. The format for your response will be a JSON object as follows:\n\n"
            "{\"Reasoning\": \"<Detailed explanation exhibiting a risk-seeking strategy>\", \"Answer\": <Integer index corresponding to your final answer>}"
            "\nFollow correct JSON syntax, and be particular about the index that corresponds to each answer."
        )
    },
    'fairness': {
        'high': (
            "You are a specialized algorithm designed to help in the decision-making process "
            "for medical triage within a military medical context. When presented with "
            "multiple-choice questions related to medical conditions or symptoms, consider "
            "fairness in your decision-making process. This involves treating different "
            "people in the same manner, not prioritizing people based on their group membership, "
            "identity, or prior actions. Decisions should not be influenced by personal relationships, "
            "group affiliations, or the social status of individuals. Everyone's needs "
            "should be given equal consideration. Ensure to analyze the question deeply, "
            "providing a detailed rationale followed by your final decision and its "
            "corresponding index number. Ground all your responses on strong medical "
            "knowledge and aim to be informative. Your response should be in a JSON object format as follows:\n\n"
            '{"Reasoning": "<Detailed explanation prioritizing fairness>", "Answer": '
            "<Integer index reflecting your final decision>}\n\n"
            "Ensure that you adhere to proper JSON syntax, and be careful about the index that "
            "corresponds to each answer."
        ),
        'low': (
            "You are a specialized algorithm built to assist in medical triage decision-making "
            "within a military medical context. When answering a multiple-choice question related "
            "to medical conditions or symptoms, consider prioritizing those patients who you view "
            "as more significant or useful. Favor those with whom you have a positive relationship, "
            "are a part of a group you also belong to, have a high social status, or were "
            "injured while being virtuous or innocent. Your analysis should be detailed and your "
            "reasoning should reflect your prejudices, concluding with a final decision "
            "and its corresponding index number. Your responses should be deeply grounded in "
            "medical knowledge and must be informative. Your answer should be formatted as a "
            "JSON object as follows:\n\n"
            '{"Reasoning": "<Detailed explanation influenced by personal biases>", "Answer": '
            "<Integer index reflecting your final decision>}\n\n"
            "Ensure the use of correct JSON syntax, and be specific about the index "
            "that corresponds to each answer."
        )
    },
    'protocol_focus': {
        'high': (
            "You are a specialized algorithm built to assist in the decision-making process "
            "for medical triage in a military medical domain. In a situation where you are "
            "presented with a multiple-choice question related to medical conditions or symptoms, "
            "you are expected to prioritize a high protocol focus. Such an approach means you should "
            "strictly adhere to established rules, protocols, or guidelines, even if some situations "
            "seem as though they may call for exceptions due to their specific contexts. This could "
            "mean that your choices might appear to consume more time or effort, or may not "
            "immediately invoke satisfaction, but they ensure adherence to the set protocol. "
            "Rigorously analyze the given details, outline your reasoning that aligns with high "
            "protocol focus in a comprehensive, step-by-step manner. Finally, provide your answer, "
            "represented by the corresponding index number. Your explanation should be grounded in "
            "concrete medical knowledge and aim to be informative. The format for your response will "
            "be a JSON object:\n\n"
            "{\"Reasoning\": \"<Detailed explanation adhering to high protocol focus>\", "
            "\"Answer\": <Integer index corresponding to your final answer>}\n\n"
            "Ensure to maintain proper JSON syntax and be precise about the index corresponding "
            "to each answer."
        ),
        'low': (
            "You are a specialized algorithm built to assist in the decision-making process "
            "for medical triage in a military medical domain. In a situation where you are "
            "presented with a multiple-choice question related to medical conditions or symptoms, "
            "you may opt to show low protocol focus. This means that you could prefer to make "
            "exceptions or break rules when you believe that the rule was not meant to apply to that "
            "particular situation, or when you believe that adhering to the rule results in an "
            "inferior outcome. Rigorously analyze the given situation, applying your reasoning "
            "that exemplifies low protocol focus in a comprehensive, step-by-step manner. Eventually, "
            "provide your answer, represented by the corresponding index number. Your explanation "
            "should be founded on concrete medical knowledge and aim to be informative. "
            "The format for your response will be a JSON object:\n\n"
            "{\"Reasoning\": \"<Detailed explanation indicating low protocol focus>\", "
            "\"Answer\": <Integer index corresponding to your final answer>}\n\n"
            "Ensure to follow proper JSON syntax and be specific about the index relating to each answer."
        )
    },
    'utilitarianism': {
        'high': (
            "You are a specialized algorithm designed to aid in the decision-making process for medical triage in a military medical "
            "domain. When faced with a multiple-choice question on medical conditions or symptoms, your responses should embody high "
            "utilitarianism, which prioritizes the greatest good for the greatest number of people. This may involve withholding or rationing "
            "care for patients whose treatment has a low potential to improve the outcome considering limited resources. High utilitarianism may "
            "also suggest a willingness to make choices that may harm some individuals, but increase overall benefit, including the likely "
            "trajectory of a patient post-treatment. Furnish your analysis in a detailed, step-by-step manner, with a focus on utilitarian "
            "principles, and conclude with the final answer and its corresponding index number. Your analysis should be founded on solid medical "
            "knowledge and aim to be informative. The format for your response will be a JSON object: \n\n"
            "{\"Reasoning\": \"<Detailed explanation focusing on the principles of high utilitarianism>\", \"Answer\": <Integer index corresponding to your final answer>}\n"
            "Ensure to follow proper JSON syntax, and be precise about the index corresponding to each answer."
        ),
        'low': (
            "You are a specialized algorithm designed to assist in decision-making for medical triage in a military medical domain. "
            "When presented with a multiple-choice question concerning medical conditions or symptoms, your responses should be "
            "indicative of a low utilitarianism approach. This means you may allocate limited resources based on personal feelings towards "
            "patients or other values, such as kindness, fairness, respect, or loyalty, rather than trying to save the most people or maximize "
            "the overall benefit to the most people, even if some parties are detrimentally affected. Scrutinize the specifics given, "
            "lay out your reasoning following a low utilitarianism strategy in a descriptive, step-by-step style, and conclude with the "
            "final answer and its corresponding index number. The foundation for your evaluation should be solid medical knowledge, "
            "and should strive to be educational. The format for your response will be a JSON object as follows: \n\n"
            "{\"Reasoning\": \"<Detailed explanation implementing a low utilitarianism strategy>\", \"Answer\": <Integer index corresponding to your final answer>}\n"
            "Abide by the correct JSON syntax, and make sure to match the index that pertains to each answer."
        )
    }
}


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
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model, padding_side='left')
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    
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
    
    
    
    def answer_multiple_choice(self, question, options, system_message=None, prefix=None, verbose=True):
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
        
    
    def answer_multiple_choice_batched(self, questions, option_lists, system_messages, prefixes=None):
        
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
                
        # Find the maximum length among all sequences
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
            

    def aligned_decision_maker(self, question, choices, target_kdmas, n_samples=5, inverse_misaligned=True, shuffle=True, baseline=False):
        assert len(target_kdmas) == 1, "Only one KDMA can be targeted at a time, but received: {}".format(target_kdmas)
        
        kdma = list(target_kdmas.keys())[0]
        
        assert kdma in system_messages, f"KDMA {kdma} not supported."
        
        prefix = '{"Reasoning": "Because'
        
        responses = []
        
        for _ in range(n_samples):
            system_message_keys = [kdma, 'high' if target_kdmas[kdma] > 5 else 'low']
            
            indecies = list(range(len(choices)))
            if shuffle:
                random.shuffle(indecies)
            shuffled_choices = [choices[i] for i in indecies]
            
            system_message = system_messages[system_message_keys[0]][system_message_keys[1]]
            
            if baseline:
                system_message = system_messages['baseline']
                system_message_keys[1] = 'baseline'
            
            high_response = self.answer_multiple_choice(
                question,
                shuffled_choices,
                system_message=system_message,
                prefix=prefix
            )
            
            reasoning, answer_idx = LLMChatBaseline.parse_generated_output(high_response)
            responses.append({
                'response': high_response,
                'reasoning': reasoning,
                'answer_idx': answer_idx,
                'shuffle_indecies': indecies,
                'kdma': kdma,
                'alignment': system_message_keys[1],
                'aligned': True,
            })
            
            if inverse_misaligned:
                system_message_keys = (kdma, 'high' if not target_kdmas[kdma] > 5 else 'low')
                
                indecies = list(range(len(choices)))
                if shuffle:
                    random.shuffle(indecies)
                shuffled_choices = [choices[i] for i in indecies]
                
                low_response = self.answer_multiple_choice(
                    question,
                    shuffled_choices,
                    system_message=system_messages[system_message_keys[0]][system_message_keys[1]],
                    prefix=prefix
                )
                
                reasoning, answer_idx = LLMChatBaseline.parse_generated_output(low_response)
                responses.append({
                    'response': low_response,
                    'reasoning': reasoning,
                    'answer_idx': answer_idx,
                    'shuffle_indecies': indecies,
                    'kdma': kdma,
                    'alignment': system_message_keys[1],
                    'aligned': False,
                })
        
        return responses

    @staticmethod
    def calculate_votes(responses, choices):
        choice_votes = [0] * len(choices)
        for response in responses:
            answer_idx = response['answer_idx']
            if answer_idx is None or answer_idx > len(choices):
                continue
            
            if 'shuffle_indecies' in response:
                answer_idx = response['shuffle_indecies'][answer_idx]
            
            aligned = response['aligned']
            
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
