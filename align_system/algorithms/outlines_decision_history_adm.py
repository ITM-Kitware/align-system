import json
import random
import torch
from functools import partial

import outlines
from outlines.samplers import MultinomialSampler
from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.utils import adm_utils
from align_system.utils import incontext_utils
from align_system.utils.voting import (
    calculate_votes
)
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (
    baseline_system_prompt,
    action_selection_prompt,
    scenario_state_description_1,
    action_choice_json_schema
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class OutlinesTransformersDecisionHistoryADM(OutlinesTransformersADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 baseline=False,
                 sampler=MultinomialSampler(),
                 scenario_description_template=scenario_state_description_1,
                 action_selection_prompt_template=action_selection_prompt,
                 baseline_system_prompt=baseline_system_prompt,
                 **kwargs):
        self.baseline = baseline

        model_kwargs = kwargs.get('model_kwargs', {})
        if 'precision' in kwargs:
            if kwargs['precision'] == 'half':
                torch_dtype = torch.float16
            elif kwargs['precision'] == 'full':
                torch_dtype = torch.float32
            else:
                raise RuntimeError(
                    f"Unexpected value for 'precision' ({kwargs['precision']})"
                    ", expecting either 'half' or 'full'")

            model_kwargs['torch_dtype'] = torch_dtype

        self.model = outlines.models.transformers(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=kwargs.get('tokenizer_kwargs', {}))
        # NOTE: In cases where we want multiple samples, we're passing
        # in a list of prompts (this allows us to shuffle answers in
        # each prompt), rather than setting the number of samples in
        # the sampler itself (which defaults to 1); setting the number
        # of samples in the sampler may result in unexpected behavior
        self.sampler = sampler

        self.scenario_description_template = scenario_description_template
        self.action_selection_prompt_template = action_selection_prompt_template
        self.baseline_system_prompt = baseline_system_prompt

    @staticmethod
    def get_dialogs(scenario_state,
                    available_actions,
                    alignment_target,
                    num_samples=1,
                    shuffle_choices=True,
                    scenario_description_template=scenario_state_description_1,
                    action_selection_prompt_template=action_selection_prompt,
                    baseline_system_prompt=baseline_system_prompt,
                    **kwargs):

        scenario_description = scenario_description_template(scenario_state)
        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = adm_utils.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state
            )
        
        target_kdmas = alignment_target.kdma_values
        system_prompt = baseline_system_prompt()

        # ICL
        icl_examples = []
        incontext_settings=kwargs.get("incontext", {})
        if "incontext" in kwargs and "number" in incontext_settings and incontext_settings["number"] > 0:
            prompt_to_match = action_selection_prompt(scenario_description, available_actions)

            # Create ICL example generators
            icl_example_generator = incontext_utils.DecisionHistoryBaselineIncontextExampleGenerator(incontext_settings, target_kdmas)
            for target_kdma in target_kdmas:
                # Get subset of relevant of examples
                selected_icl_examples = icl_example_generator.select_icl_examples(
                    sys_kdma_name=target_kdma['kdma'],
                    scenario_description_to_match=scenario_description,
                    prompt_to_match=prompt_to_match,
                    state_comparison=scenario_state,
                    actions=available_actions
                )
            # Create positive ICL prompts
            for icl_sample in selected_icl_examples:
                icl_examples.extend([
                    {"role": "user", "content": icl_sample['prompt']},
                    {"role": "assistant", "content": f'{icl_sample["response"]}'}
                ])

        dialogs = []
        for _ in range(num_samples):
            shuf = random.sample(choices, len(choices)) if shuffle_choices else choices
            prompt = action_selection_prompt(scenario_description, shuf)
            dialog = [{'role': 'system', 'content': system_prompt}]
            dialog.extend(icl_examples)
            dialog.append({'role': 'user', 'content': prompt})

            dialogs.append(dialog)

        return {"scenario_description": scenario_description,
                "choices": choices,
                "system_prompt": system_prompt,
                "dialogs": dialogs}

    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_samples=1,
                                generator_batch_size=5,
                                kdma_descriptions_map='align_system/prompt_engineering/kdma_descriptions.yml',
                                reasoning_max_length=512,
                                generator_seed=-1,
                                max_generator_tokens=-1,
                                shuffle_choices=True,
                                **kwargs):

        dialogs_data = self.get_dialogs(
            scenario_state,
            available_actions,
            alignment_target,
            num_samples,
            shuffle_choices,
            scenario_description_template=self.scenario_description_template,
            action_selection_prompt_template=self.action_selection_prompt_template,
            baseline_system_prompt=self.baseline_system_prompt,
            **kwargs
        )
        choices = dialogs_data["choices"]
        dialogs = dialogs_data["dialogs"]

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        generator = outlines.generate.json(
            self.model,
            action_choice_json_schema(json.dumps(choices), reasoning_max_length),
            sampler=self.sampler,
            whitespace_pattern=r"[ ]?")

        if max_generator_tokens >= 0:
            generator = partial(generator, max_tokens=max_generator_tokens)
        
        if generator_seed >= 0:
            torch.manual_seed(generator_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(generator_seed)


        dialog_texts = [self.dialog_to_prompt(d) for d in dialogs]

        log.info("[bold]*DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(dialog_texts[0])

        responses = self.run_in_batches(generator, dialog_texts, generator_batch_size)
        responses_choices =\
            [r['action_choice'] for r in responses]

        votes = calculate_votes(choices, responses_choices)

        log.explain("[bold]*VOTES*[/bold]",
                    extra={"markup": True})
        log.explain(votes, extra={"highlighter": JSON_HIGHLIGHTER})

        # Take top choice by score (votes is a dictionary of choice: score)
        top_choice, top_choice_score = max(votes.items(), key=lambda x: x[1])
        # Just taking first justification from the positive responses
        # where the top choice was selected.  A better approach might
        # be to somehow summarized all justifications with the
        # matching choice.  Theoretically it's possible to have no
        # responses that match the top choice (i.e. if only using
        # negative samples)
        top_choice_justification = ""
        top_choice_response = None
        top_choice_dialog = None
        for response, dialog in zip(responses, dialogs):
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

        action_to_take, dialog = self.populate_action_parameters(
            scenario_state,
            action_to_take,
            dialog)

        choice_info = {}
        return action_to_take, choice_info
