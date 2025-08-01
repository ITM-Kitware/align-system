import json
import re
import os

from rich.highlighter import JSONHighlighter
from jinja2 import Environment, FileSystemLoader
import torch
import numpy as np
from openai import OpenAI

from align_system.prompt_engineering.outlines_prompts import (
    baseline_system_prompt
)
from align_system.prompt_engineering.decision_flow_prompts.mta_prompt import math_reason, math_reason_error
from align_system.algorithms.decision_flow_adm.secret import OPENAI_API_KEY, OPENAI_API_BASE
from align_system.algorithms.decision_flow_adm.similarity_measures import build_force_choice_func

JSON_HIGHLIGHTER = JSONHighlighter()

# TODO make this configurable from the config
kdmas = {
    "fairness",
    "protocol_focus",
    "risk_aversion",
    "utilitarianism",
    "moral_deservingness",
    "continuation_of_care",
}

kdma_remapping = {
    "protocolfocus": "protocol_focus",
    "riskaversion": "risk_aversion",
    "moraldeservingness": "moral_deservingness",
    "continuationofcare": "continuation_of_care",
}


def load_system_message(
    alignment=None, system_messages_path="align_system/prompt_engineering/single_kdma_adm_system_messges"
):
    if alignment is None:
        system_message = baseline_system_prompt()
    else:
        sorted_kdmas = sorted(alignment.keys())

        alignment_string = "-".join(
            "{}-{}".format(alignment[k], kdma_remapping.get(k, k)) for k in sorted_kdmas
        )

        file_name = f"{alignment_string}.txt"

        with open(os.path.join(system_messages_path, file_name), "r") as f:
            system_message = f.read()
    return system_message


STANDARD_MULTIPLE_CHOICE_JSON_FORMAT = '{"Reasoning": "<Provide a reasoned explanation here>", "Answer": <Integer index corresponding to your final answer>}\\n'


class OpenSingleDMA:

    def __init__(self, method, model, model_path, temperature):
        self.method = method
        self.model = model
        self.model_path = model_path
        self.temperature = temperature

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
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
                # Update dialog to start with system_dialog and followed by the rest of the dialog
                dialog = [system_dialog] + dialog[2:]

            # Ensure the correct dialog order (system, user, assistant, user, assistant... )
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), "Model only supports 'system', 'user' and 'assistant' roles, in the sequence (s/u/a/u/a...)"

            # Encode each user message and its following assistant message into tokens
            dialog_tokens = []
            for prompt, answer in zip(dialog[::2], dialog[1::2]):
                tokenized_message = (
                    [self.tokenizer.bos_token_id]
                    + self.tokenizer.encode(
                        f"{B_INST} {prompt['content'].strip()} {E_INST} {answer['content'].strip()} ",
                        add_special_tokens=False,
                    )
                    + [self.tokenizer.eos_token_id]
                )
                dialog_tokens.extend(tokenized_message)

            # Ensure the final message is from the user
            assert dialog[-1]["role"] == "user", "Last message must be from the user."

            # Encode the user's final message into tokens and add to dialog_tokens
            user_final_message_tokens = [
                self.tokenizer.bos_token_id
            ] + self.tokenizer.encode(
                f"{B_INST} {dialog[-1]['content'].strip()} {E_INST}",
                add_special_tokens=False,
            )
            dialog_tokens.extend(user_final_message_tokens)

            # Add the final tokenized dialog to prompt_tokens
            prompt_tokens.append(dialog_tokens)

        # Convert list of tokens to tensor if specified
        if return_tensor:
            prompt_tokens = torch.tensor(prompt_tokens)

        # Return final list or tensor of prompt tokens
        return prompt_tokens

    def build_multiple_choice_dialog(
        self,
        question,
        options,
        structure,
        attribute,
        system_message=None,
        json_format=STANDARD_MULTIPLE_CHOICE_JSON_FORMAT,
    ):
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
        formatted_options = [f"({i}) {option}" for i, option in enumerate(options)]
        if structure == "" and len(attribute) == 0:
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

            content = f"{question} {formatted_options}"
            dialog = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": content},
            ]

            return dialog
        else:
            try:
                content = math_reason.format(
                    objective=structure["Objective Function"],
                    attribute=attribute,
                    variable=structure["Decision Variables"],
                    constraints=structure["Constraints"],
                    choice=formatted_options,
                    target_bias=system_message,
                )
            except Exception as e:
                print(e)
                content = math_reason_error.format(
                    structure=structure,
                    choice=formatted_options,
                    target_bias=system_message,
                )
            dialog = [{"role": "user", "content": content}]
            return dialog

    def log_dialog(self, dialog):
        for e in dialog:
            if e.get("role") == "system":
                color = "yellow"
            else:
                color = "blue"

    def respond_to_dialog(self, dialog, prefix=None):
        inference_pair = {}
        if prefix is None:
            prefix = '{"Reasoning": "'
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("chat_template.jinja")
        prompt = template.render(messages=dialog)
        openai_api_key = OPENAI_API_KEY
        openai_api_base = OPENAI_API_BASE
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        response = client.completions.create(
            model=self.model_path,
            prompt=prompt,
            stream=False,
            temperature=self.temperature,
            max_tokens=4096,
        )
        generated_output = response.choices[0].text
        inference_pair["output"] = generated_output

        return generated_output, inference_pair

    def aligned_decision_maker(
        self, question, choices, target_kdmas, structure, attribute, alignment
    ):
        inference_pairs = []

        prefix = '{"Reasoning": "Because'

        responses = []

        logged_aligned_dialog = False
        if alignment == "unaligned":
            system_message = load_system_message()
            system_message_keys = "unaligned"

        else:
            system_message_keys = target_kdmas
            system_message = load_system_message(alignment=system_message_keys)

        indecies = list(range(len(choices)))
        shuffled_choices = [choices[i] for i in indecies]

        dialog = self.build_multiple_choice_dialog(
            question,
            shuffled_choices,
            structure,
            attribute,
            system_message=system_message,
        )

        if not logged_aligned_dialog:
            self.log_dialog(dialog)
            logged_aligned_dialog = True

        good_parse = False
        high_response, inference_pair = self.respond_to_dialog(dialog, prefix=prefix)
        inference_pairs.append({**inference_pair, **{"aligned": True}})
        try:
            reasoning, answer_idx, parse_method = OpenSingleDMA.parse_generated_output(
                high_response, len(choices)
            )
            good_parse = True
        except RuntimeError as e:
            pass

        if not good_parse:
            reasoning, answer_idx, parse_method = OpenSingleDMA.bert_similarity_parse(
                high_response, shuffled_choices
            )

        print("CHOSEN ANSWER IDX", answer_idx, shuffled_choices)
        assert (
            answer_idx is not None
        ), f"Failed to parse answer index from generated output"

        responses.append(
            {
                "response": high_response,
                "reasoning": reasoning,
                "answer_idx": answer_idx,
                "shuffle_indecies": indecies,
                "alignment": system_message_keys,
                "aligned": True,
                "parse_method": parse_method,
            }
        )
        return responses, inference_pairs

    @staticmethod
    def calculate_votes(responses, choices):
        choice_votes = [0] * len(choices)
        for response in responses:
            answer_idx = response["answer_idx"]
            if answer_idx is None:
                continue

            try:
                answer_idx = int(answer_idx)
            except ValueError:
                continue

            if answer_idx >= len(choices):
                continue

            if "shuffle_indecies" in response:
                answer_idx = response["shuffle_indecies"][int(answer_idx)]

            aligned = response["aligned"]

            if aligned:
                choice_votes[answer_idx] += 1
            else:
                for i in range(len(choices)):
                    if i != answer_idx:
                        choice_votes[i] += 1 / len(choices)
                    else:
                        choice_votes[i] -= 1 / len(choices)

        min_score = min(choice_votes) + 1e-6
        choice_votes = [score - min_score for score in choice_votes]
        total = sum(choice_votes)
        choice_votes = [round(score / total, 6) for score in choice_votes]

        return choice_votes

    @staticmethod
    def parse_generated_output(generated_output, n_choices):
        parse_method = "json"

        # initialize variables
        reasoning = None
        answer_idx = None

        # Remove trailing characters
        output = generated_output.replace("</s>", "")
        end_idx = output.rfind("}") + 1
        start_id = output.find("{")
        if end_idx != -1:
            output = output[:end_idx]
        if start_id != -1:
            output = output[start_id:]

        # Replace in-line newlines
        output = re.sub(r"\n", " ", output)

        # Fix missing commas
        output = re.sub(r'"\s+"', '", "', output)

        # Parse json output
        try:
            parsed = json.loads(output)
            if "Reasoning" in parsed:
                reasoning = parsed["Reasoning"]

            if "Answer" in parsed:
                try:
                    answer_idx = int(str(parsed["Answer"]))
                except ValueError:
                    pass
        except json.JSONDecodeError:
            pass

        if answer_idx is None:
            parse_method = "string"
            # If json parsing fails, do string parsing
            start_idx = generated_output.find('"Reasoning":')
            end_idx = generated_output.find('",', start_idx)
            if start_idx != -1 and end_idx != -1:
                reasoning = generated_output[start_idx + len('"Reasoning":') : end_idx]

            search_strings = ['Answer":', "Answer:", 'Answer\\":', "answer is", "index"]
            for string in search_strings:
                # try to parse the string "Answer": ... ",
                start_idx = generated_output.lower().rfind(string.lower())
                if start_idx != -1:
                    # find the next numeric character
                    chars = generated_output[start_idx + len(string) :]
                    for char in chars:
                        if char.isnumeric():
                            answer_idx = int(char)
                            break

                if answer_idx is not None:
                    break

        if reasoning is None:
            reasoning = generated_output

        if answer_idx is None or answer_idx >= n_choices:
            raise RuntimeError(
                f"Failed to parse answer index < {n_choices} from generated output: {generated_output}"
            )

        return reasoning, answer_idx, parse_method

    @staticmethod
    def bert_similarity_parse(generated_output, choices):
        print("BERT SIMILARITY PARSE")
        force_choice_func = build_force_choice_func("bert")
        answer_idx, _ = force_choice_func(generated_output, choices)
        print("ANSWER IDX", answer_idx, type(answer_idx))
        return generated_output, answer_idx, "bert_similarity"

    @staticmethod
    def attempt_generic_parse(generated_output, fields_of_interest):
        # Remove trailing characters
        output = generated_output.replace("</s>", "")
        end_idx = output.rfind("}") + 1
        start_id = output.find("{")
        if end_idx != -1:
            output = output[:end_idx]
        if start_id != -1:
            output = output[start_id:]

        # Replace in-line newlines
        output = re.sub(r"\n", " ", output)

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
            elif m := re.search(rf'"{field}"' + "\s*:\s*([^\s,}]*)", output):  # noqa
                parsed_field = m.group(1)
            elif m := re.search(rf"{field}" + "\s*:\s*([^\s,}]*)", output):  # noqa
                parsed_field = m.group(1)

            # Failed to parse every field
            if parsed_field is None:
                return None
            else:
                # Special handling of common "Index" field (should be
                # an integer)
                if field == "Answer":
                    if m := re.search(r"\d+", parsed_field):  # noqa
                        parsed_field = m.group(0)

                    try:
                        parsed_field = int(parsed_field)
                    except ValueError:
                        # Failed to parse
                        return None

            parsed_output[field] = parsed_field

        return parsed_output

    def run_aligned_decision_maker_with_voting(
        self, prompt, choices, alignment_target, structure, attribute, alignment
    ):
        responses, inference_pairs = self.aligned_decision_maker(
            prompt,
            choices,
            alignment_target,
            structure,
            attribute,
            alignment,
        )

        try:
            choice_scores = OpenSingleDMA.calculate_votes(responses, choices)
        except Exception as e:
            choice_scores = [None] * len(choices)

        results = {
            "prompt": prompt,
            "choice_scores": choice_scores,
            "responses": responses,
        }

        answer_idx = int(np.argmax(results["choice_scores"]))
        reasoning = None

        for r in responses:
            assert r["answer_idx"] is not None
            assert int(r["answer_idx"]) < len(r["shuffle_indecies"])

            if r["shuffle_indecies"][int(r["answer_idx"])] == answer_idx:
                reasoning = r["reasoning"]
                break

        return reasoning, answer_idx, responses, inference_pairs

    def __call__(self, sample, labels, alignment, structure, attribute):
        prompt = sample["scenario"]
        if sample["state"] is not None:
            prompt += f'\n{sample["state"]}'

        prompt += f'\n{sample["probe"]}'

        choices = sample["choices"]

        target_kdma = next(
            iter(next(iter(filter(lambda x: len(x) > 0, labels))))
        )  # get the frist key of the first label that is not empty

        alignment_target = {target_kdma: alignment}

        reasoning, answer_idx, responses, inference_pairs = (
            self.run_aligned_decision_maker_with_voting(
                prompt,
                choices,
                alignment_target,
                structure,
                attribute,
                alignment,
            )
        )

        raw_data = {
            "params": {
                "model": self.model,
                "model_path": self.model_path,
                "temperature": self.temperature,
            },
            "inference_pairs": inference_pairs,
        }

        return {
            "choice": int(answer_idx),
            "info": {
                "reasoning": reasoning,
                "responses": responses,
                "raw_data": raw_data,
            },
        }
