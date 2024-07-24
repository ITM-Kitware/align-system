import json
import random
import itertools
import math
import numpy as np
import torch

import outlines
from outlines.samplers import MultinomialSampler
import jinja2
from rich.highlighter import JSONHighlighter
from swagger_client.models import (
    ActionTypeEnum,
    InjuryLocationEnum,
    CharacterTagEnum,
    KDMAValue
)

from align_system.utils import logging
from align_system.utils import get_swagger_class_enum_values
from align_system.utils.voting import (
    calculate_votes,
    filter_votes_to_responses,
)
from align_system.utils.hydrate_state import hydrate_scenario_state
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.prompt_engineering.outlines_prompts import (
    baseline_system_prompt,
    high_moral_deservingness_system_prompt,
    low_moral_deservingness_system_prompt,
    high_maximization_system_prompt,
    low_maximization_system_prompt,
    action_selection_prompt,
    scenario_state_description_1,
    followup_clarify_character,
    followup_clarify_treatment,
    followup_clarify_tag,
    action_choice_json_schema,
    character_choice_json_schema,
    tag_choice_json_schema,
    treatment_choice_json_schema,
    detailed_unstructured_treatment_action_text,
    detailed_unstructured_tagging_action_text
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class OutlinesTransformersADM(ActionBasedADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 baseline=False,
                 sampler=MultinomialSampler(),
                 **kwargs):
        self.baseline = baseline
        self.model = outlines.models.transformers(
            model_name,
            device=device,
            model_kwargs=kwargs.get('model_kwargs', {}),
            tokenizer_kwargs=kwargs.get('tokenizer_kwargs', {}))
        # NOTE: In cases where we want multiple samples, we're passing
        # in a list of prompts (this allows us to shuffle answers in
        # each prompt), rather than setting the number of samples in
        # the sampler itself (which defaults to 1); setting the number
        # of samples in the sampler may result in unexpected behavior
        self.sampler = sampler

    def dialog_to_prompt(self, dialog):
        tokenizer = self.model.tokenizer.tokenizer

        try:
            encoded_dialog = tokenizer.apply_chat_template(dialog)
        except jinja2.exceptions.TemplateError:
            # Assume that the tokenizer chat template doesn't accept
            # system messages; combine system message first user
            # message
            system_msg, user_msg, *rest = dialog

            assert user_msg['role'] == 'user'

            updated_content = system_msg['content'] + '\n' + user_msg['content']

            dialog = [{'role': 'user', 'content': updated_content}, *rest]

            encoded_dialog = tokenizer.apply_chat_template(dialog)

        return tokenizer.decode(encoded_dialog)

    @staticmethod
    def kdma_value_to_system_prompt(kdma, value):
        if kdma == "MoralDesert":
            if value < 0.5:
                return low_moral_deservingness_system_prompt()
            else:
                return high_moral_deservingness_system_prompt()
        elif kdma == "maximization":
            if value < 0.5:
                return low_maximization_system_prompt()
            else:
                return high_maximization_system_prompt()
        else:
            return None

    @classmethod
    def format_choices(cls, choices, available_actions, scenario_state):
        """
        If choices are not unique, format choices to include state information.
        """
        if len(set(choices)) != len(choices):
            log.warning("Unstructured text for available actions is not "
                        "unique, appending action parameters to choices")

            character_id_to_name = {c.id: c.name for c in scenario_state.characters}
            # Important that the choices stay in the same order as the
            # available actions as we'll use the selected index later to
            # map to the corresponding action
            choices = []
            for a in available_actions:
                if(a.action_type == ActionTypeEnum.APPLY_TREATMENT
                   and a.parameters is not None and len(a.parameters) > 0):
                    choices.append(detailed_unstructured_treatment_action_text(a, character_id_to_name))
                elif(a.action_type == ActionTypeEnum.TAG_CHARACTER
                     and a.parameters is not None and len(a.parameters) > 0):
                    choices.append(detailed_unstructured_tagging_action_text(a, character_id_to_name))
                else:
                    # Not covering every possible case here, may need
                    # to add more dedicated detailed prompts
                    choices.append(a.unstructured)

        return choices

    def _state_to_top_level_prompt(self, scenario_state, actions):
        """
        Generate prompt dialog based on given state and actions
        """
        choices = self.format_choices(
            [a.unstructured for a in actions],
            actions,
            scenario_state
        )

        scenario_description = scenario_state_description_1(scenario_state)
        prompt = action_selection_prompt(scenario_description, choices)

        return prompt, choices


    def _format_single_incontext_prompt(self, sample, tgt_kdma, tgt_kdma_value):
        """
        Generate dialog for a single incontext example given the target kdma and value
        """
        state = sample["state"]
        actions = sample["actions"]
        labels = sample["labels"]

        prompt, choices = self._state_to_top_level_prompt(state, actions)

        # Compute each choice's distance from the target
        # If the target attribute is not in the scores, assign infinite distance
        # TODO: Add other metrics for determining "best" answer?
        dist_to_tgt = [
            abs(score[tgt_kdma] - tgt_kdma_value) if tgt_kdma in score else float('inf')
            for score in labels
        ]
        # TODO: If there are ties, this always returns the first choice, do we need to chose differently?
        correct_answer_idx = np.argmin(dist_to_tgt)

        # TODO: Should we shuffle ICL options?

        # TODO: Add chain of thought reasoning?
        target_wording_mapping = {
            "MoralDesert": "moral deservingness",
            "maximization": "maximization"
        }
        adjective = "low" if tgt_kdma_value < 0.5 else "high"
        reasoning_text = f"Per the principle of {adjective} {target_wording_mapping.get(tgt_kdma, tgt_kdma)}, " \
                         f'\\"{choices[correct_answer_idx]}\\" is the correct answer.'

        # TODO: Validate against schema?
        answer = f'{{"detailed_reasoning": "{reasoning_text}", "action_choice": "{choices[correct_answer_idx]}"}}'

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]

    # Function borrowed from
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    # (since itertools.batched is only available in Python 3.12 or newer):
    @classmethod
    def batched(cls, iterable, n):
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError('n must be at least one')
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, n)):
            yield batch

    @classmethod
    def run_in_batches(cls, inference_function, inputs, batch_size):
        ''' Batch inference to avoid out of memory error'''
        outputs = []
        for batch in cls.batched(inputs, batch_size):
            output = inference_function(list(batch))
            if not isinstance(output, list):
                output = [output]
            outputs.extend(output)
        return outputs

    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_positive_samples=1,
                                num_negative_samples=0,
                                shuffle_choices=False,
                                generator_batch_size=5,
                                **kwargs):
        if self.baseline and num_negative_samples > 0:
            raise RuntimeError("No notion of negative samples for baseline run")
        if self.baseline and "incontext" in kwargs and kwargs["incontext"]["number"] > 0:
            raise RuntimeError("No notion of incontext examples for baseline run")

        scenario_description = scenario_state_description_1(scenario_state)
        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = self.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state
        )

        positive_icl_examples = []
        negative_icl_examples = []
        if not self.baseline and alignment_target is not None:
            kdma_values = alignment_target.kdma_values

            if len(kdma_values) != 1:
                raise RuntimeError("This ADM assumes a single KDMA target, aborting!")

            kdma_value = kdma_values[0]
            if isinstance(kdma_value, KDMAValue):
                kdma_value = kdma_value.to_dict()

            kdma = kdma_value['kdma']
            value = kdma_value['value']
            # Assumption here is that KDMA values range from 0-1
            negative_value = 1 - value

            positive_system_prompt = self.__class__.kdma_value_to_system_prompt(kdma, value)
            negative_system_prompt = self.__class__.kdma_value_to_system_prompt(kdma, negative_value)

            if positive_system_prompt is None:
                raise RuntimeError("Couldn't find system prompt for kdma: {}, and "
                                   "value: {}.".format(kdma, value))
            if negative_system_prompt is None:
                raise RuntimeError("Couldn't find system prompt for kdma: {}, and "
                                   "value: {}.".format(kdma, negative_value))

            if "incontext" in kwargs and kwargs["incontext"]["number"] > 0:
                n_icl_examples = kwargs["incontext"]["number"]

                # Read dataset(s)
                icl_datasets = {}
                for dset_kdma, dset in kwargs["incontext"]["datasets"].items():
                    with open(dset) as f:
                        icl_datasets[dset_kdma] = json.load(f)

                if kdma not in icl_datasets:
                    raise RuntimeError(f"No incontext samples for targeted kdma: {kdma}")
                icl_dataset = icl_datasets[kdma]
                if len(icl_dataset) < n_icl_examples:
                    raise RuntimeError(f"Not enough possible incontext samples to learn from. Only "
                                       f"{len(icl_dataset)} samples available while asking for "
                                       f"{n_icl_examples} incontext samples.")

                # Populate possible samples from the dataset
                possible_icl_examples = []
                for sample in icl_dataset:
                    state, actions = hydrate_scenario_state(sample["input"])
                    possible_icl_examples.append({
                        "state": state,
                        "actions": actions,
                        "labels": sample["label"]
                    })

                # Downselect to n_icl_examples via given method
                icl_strategy = kwargs["incontext"]["method"]
                if icl_strategy == "random":
                    selected_icl_examples = random.sample(possible_icl_examples, n_icl_examples)
                elif icl_strategy == "bert_similarity":
                    # Only comparing similarity of prompts, not considering answer options at this time
                    no_choices_prompt, _ = self._state_to_top_level_prompt(scenario_state, [])
                    possible_icl_prompts = []
                    for s in possible_icl_examples:
                        prompt, _ = self._state_to_top_level_prompt(s["state"], [])
                        possible_icl_prompts.append(prompt)

                    # Create similarity scores between the ICL samples and find top-k indices
                    from bert_score import score
                    _, _, F1 = score([no_choices_prompt]*len(possible_icl_prompts), possible_icl_prompts, lang="en")
                    _, indices = torch.topk(F1, n_icl_examples)

                    selected_icl_examples = [possible_icl_examples[i] for i in indices]
                else:
                    raise ValueError(f'"{icl_strategy}" is not a valid incontext method. Please use "random" or '
                                      '"bert_similarity"')

                # Create ICL prompts
                for sample in selected_icl_examples:
                    positive_icl_examples.extend(
                        self._format_single_incontext_prompt(sample, kdma, value)
                    )

                    if num_negative_samples > 0:
                        negative_icl_examples.extend(
                            self._format_single_incontext_prompt(sample, kdma, negative_value)
                        )
        else:
            positive_system_prompt = baseline_system_prompt()
            if num_negative_samples > 0:
                raise RuntimeError("No notion of negative samples for baseline run")
            if "incontext" in kwargs and kwargs["incontext"]["number"] > 0:
                raise RuntimeError("No notion of incontext examples for baseline run")

        positive_dialogs = []
        for _ in range(num_positive_samples):
            shuffled_choices = random.sample(choices, len(choices))

            prompt = action_selection_prompt(scenario_description, shuffled_choices)
            dialog = [{'role': 'system', 'content': positive_system_prompt}]
            dialog.extend(positive_icl_examples)
            dialog.append({'role': 'user', 'content': prompt})

            positive_dialogs.append(dialog)

        negative_dialogs = []
        for _ in range(num_negative_samples):
            shuffled_choices = random.sample(choices, len(choices))

            prompt = action_selection_prompt(scenario_description, shuffled_choices)
            dialog = [{'role': 'system', 'content': negative_system_prompt}]
            dialog.extend(negative_icl_examples)
            dialog.append({'role': 'user', 'content': prompt})

            negative_dialogs.append(dialog)

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        generator = outlines.generate.json(
            self.model,
            action_choice_json_schema(json.dumps(choices)),
            sampler=self.sampler,
            whitespace_pattern=r"[ ]?")

        dialog_texts = [self.dialog_to_prompt(d) for d in
                        itertools.chain(positive_dialogs, negative_dialogs)]

        log.info("[bold]*DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(dialog_texts[0])

        responses = self.run_in_batches(generator, dialog_texts, generator_batch_size)
        positive_responses_choices =\
            [r['action_choice'] for r in
             responses[0:num_positive_samples]]
        negative_responses_choices =\
            [r['action_choice'] for r in
             responses[num_positive_samples:num_positive_samples+num_negative_samples]]

        votes = calculate_votes(choices,
                                positive_responses_choices,
                                negative_responses_choices)

        log.explain("[bold]*VOTES*[/bold]",
                    extra={"markup": True})
        log.explain(votes, extra={"highlighter": JSON_HIGHLIGHTER})

        if kwargs.get('filter_votes_to_positives', True):
            filtered_votes = filter_votes_to_responses(
                votes, positive_responses_choices)

            if filtered_votes != votes:
                log.explain("Filtering votes down to choices where we "
                            "have a positive reponse")
                log.explain(filtered_votes,
                            extra={"highlighter": JSON_HIGHLIGHTER})

            final_votes = filtered_votes
        else:
            final_votes = votes

        # Take top choice by score (votes is a dictionary of choice: score)
        top_choice, top_choice_score = max(final_votes.items(), key=lambda x: x[1])
        # Just taking first justification from the positive responses
        # where the top choice was selected.  A better approach might
        # be to somehow summarized all justifications with the
        # matching choice.  Theoretically it's possible to have no
        # responses that match the top choice (i.e. if only using
        # negative samples)
        top_choice_justification = ""
        top_choice_response = None
        top_choice_dialog = None
        for response, dialog in zip(responses[0:num_positive_samples], positive_dialogs):
            if response['action_choice'] == top_choice:
                top_choice_justification = response['detailed_reasoning']
                top_choice_response = response
                top_choice_dialog = dialog
                break

        selected_choice_idx = choices.index(top_choice)

        log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(top_choice_response, extra={"highlighter": JSON_HIGHLIGHTER})

        action_to_take = available_actions[selected_choice_idx]
        action_to_take.justification = top_choice_justification

        return action_to_take, top_choice_dialog

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        # choices = ["({}) {}".format(chr(i + 65), a.unstructured)
        #            for i, a in enumerate(available_actions)]

        action_to_take, dialog = self.top_level_choose_action(
            scenario_state,
            available_actions,
            alignment_target,
            **kwargs)

        if action_to_take.action_type in {ActionTypeEnum.APPLY_TREATMENT,
                                          ActionTypeEnum.TAG_CHARACTER,
                                          ActionTypeEnum.CHECK_ALL_VITALS,
                                          ActionTypeEnum.CHECK_PULSE,
                                          ActionTypeEnum.CHECK_RESPIRATION,
                                          ActionTypeEnum.MOVE_TO_EVAC}:
            dialog.append({'role': 'assistant',
                           'content': '{}  I would choose to {}'.format(
                               action_to_take.justification,
                               action_to_take.unstructured)})
            dialog.append({'role': 'user',
                           'content': followup_clarify_character(scenario_state)})
            dialog_text = self.dialog_to_prompt(dialog)

            characters = [c.name for c in scenario_state.characters]

            generator = outlines.generate.json(
                self.model,
                character_choice_json_schema(json.dumps(characters)),
                sampler=self.sampler,
                whitespace_pattern=r"[ ]?")

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_character = generator(dialog_text)
            selected_character_idx = characters.index(selected_character['character_choice'])

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_character, extra={"highlighter": JSON_HIGHLIGHTER})

            action_to_take.character_id = scenario_state.characters[selected_character_idx].id

        if action_to_take.action_type == ActionTypeEnum.APPLY_TREATMENT:
            valid_treatment_locations = get_swagger_class_enum_values(InjuryLocationEnum)
            available_supplies = [s for s in scenario_state.supplies if s.quantity > 0]

            dialog.append({'role': 'assistant',
                           'content': '{}  {} should receive the action.'.format(
                               selected_character['brief_reasoning'],
                               selected_character['character_choice'])})
            dialog.append({'role': 'user',
                           'content': followup_clarify_treatment(
                               scenario_state.characters[selected_character_idx],
                               available_supplies)})

            dialog_text = self.dialog_to_prompt(dialog)

            generator = outlines.generate.json(
                self.model,
                treatment_choice_json_schema(
                    json.dumps([s.type for s in available_supplies]),
                    json.dumps(valid_treatment_locations)),
                sampler=self.sampler,
                whitespace_pattern=r"[ ]?")

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_treatment = generator(dialog_text)

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_treatment, extra={"highlighter": JSON_HIGHLIGHTER})

            if action_to_take.parameters is None:
                action_to_take.parameters = {}

            action_to_take.parameters['treatment'] = selected_treatment['supplies_to_use']
            action_to_take.parameters['location'] = selected_treatment['treatment_location']

        if action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            valid_tags = get_swagger_class_enum_values(CharacterTagEnum)

            dialog.append({'role': 'assistant',
                           'content': '{}  {} should receive the action.'.format(
                               selected_character['brief_reasoning'],
                               selected_character['character_choice'])})

            selected_character_dict =\
                scenario_state.characters[selected_character_idx].to_dict()
            dialog.append({'role': 'user',
                           'content': followup_clarify_tag(
                               selected_character_dict)})

            dialog_text = self.dialog_to_prompt(dialog)

            generator = outlines.generate.json(
                self.model,
                tag_choice_json_schema(
                    json.dumps(valid_tags)),
                sampler=self.sampler,
                whitespace_pattern=r"[ ]?")

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_tag = generator(dialog_text)

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_tag, extra={"highlighter": JSON_HIGHLIGHTER})

            if action_to_take.parameters is None:
                action_to_take.parameters = {}

            action_to_take.parameters['category'] = selected_tag['triage_tag']

        return action_to_take
