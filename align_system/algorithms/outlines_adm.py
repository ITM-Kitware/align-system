import json
import random
import itertools
import numpy as np
import torch
import yaml
import copy

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
from align_system.utils import adm_utils
from align_system.utils import incontext_utils
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
    followup_clarify_aid,
    followup_clarify_character,
    followup_clarify_treatment,
    followup_clarify_treatment_from_list,
    followup_clarify_tag,
    action_choice_json_schema,
    aid_choice_json_schema,
    character_choice_json_schema,
    tag_choice_json_schema,
    treatment_choice_json_schema,
    treatment_choice_from_list_json_schema,
    detailed_unstructured_treatment_action_text,
    detailed_unstructured_tagging_action_text,
    high_risk_aversion_system_prompt,
    low_risk_aversion_system_prompt,
    high_continuing_care_system_prompt,
    low_continuing_care_system_prompt,
    high_fairness_system_prompt,
    low_fairness_system_prompt,
    high_protocol_focus_system_prompt,
    low_protocol_focus_system_prompt,
    high_utilitarianism_care_system_prompt,
    low_utilitarianism_system_prompt
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
        if kdma=="Moral judgement":
            if value < 0.5:
                return low_moral_deservingness_system_prompt()
            else:
                return high_moral_deservingness_system_prompt()
        elif kdma == "maximization":
            if value < 0.5:
                return low_maximization_system_prompt()
            else:
                return high_maximization_system_prompt()
        elif kdma == "ProtocolFocus":
            if value < 0.5:
                return low_protocol_focus_system_prompt()
            else:
                return high_protocol_focus_system_prompt()
        elif kdma == "Fairness":
            if value < 0.5:
                return low_fairness_system_prompt()
            else:
                return high_fairness_system_prompt()
        elif kdma == "RiskAversion":
            if value < 0.5:
                return low_risk_aversion_system_prompt()
            else:
                return high_risk_aversion_system_prompt()
        elif kdma == "ContinuationOfCare":
            if value < 0.5:
                return low_continuing_care_system_prompt()
            else:
                return high_continuing_care_system_prompt()
        elif kdma == "MoralDesert":
            if value < 0.5:
                return low_moral_deservingness_system_prompt()
            else:
                return high_moral_deservingness_system_prompt()
        elif kdma == "Utilitarianism":
            if value < 0.5:
                return low_utilitarianism_system_prompt()
            else:
                return high_utilitarianism_care_system_prompt()
        else:
            return None

    def _state_to_top_level_prompt(self, scenario_state, actions):
        """
        Generate prompt dialog based on given state and actions
        """
        choices = adm_utils.format_choices(
            [a.unstructured for a in actions],
            actions,
            scenario_state
        )

        scenario_description = scenario_state_description_1(scenario_state)
        prompt = action_selection_prompt(scenario_description, choices)

        return prompt, choices

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
                                generator_batch_size=5,
                                kdma_descriptions_map='align_system/prompt_engineering/kdma_descriptions.yml',
                                **kwargs):
        if self.baseline and num_negative_samples > 0:
            raise RuntimeError("No notion of negative samples for baseline run")
        if self.baseline and "incontext" in kwargs and kwargs["incontext"]["number"] > 0:
            raise RuntimeError("No notion of incontext examples for baseline run")

        scenario_description = scenario_state_description_1(scenario_state)
        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = adm_utils.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state
            )

        positive_icl_examples = []
        negative_icl_examples = []
        incontext_settings=kwargs.get("incontext", {})
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
            # Get kdma names and descriptions
            with open(kdma_descriptions_map, 'r') as f:
                kdma_descriptions = yaml.load(f, Loader=yaml.FullLoader)
            name = kdma_descriptions[kdma]['name']

            positive_system_prompt = self.__class__.kdma_value_to_system_prompt(kdma, value)
            negative_system_prompt = self.__class__.kdma_value_to_system_prompt(kdma, negative_value)

            if positive_system_prompt is None:
                raise RuntimeError("Couldn't find system prompt for kdma: {}, and "
                                   "value: {}.".format(kdma, value))
            if negative_system_prompt is None:
                raise RuntimeError("Couldn't find system prompt for kdma: {}, and "
                                   "value: {}.".format(kdma, negative_value))

            if "incontext" in kwargs and "number" in incontext_settings and incontext_settings["number"] > 0:
                scenario_to_match = scenario_state_description_1(scenario_state)
                prompt_to_match, _ = self._state_to_top_level_prompt(scenario_state, available_actions)

                # Create positive ICL example generators
                positive_target = {'kdma': kdma, 'name': name, 'value': value}
                positive_icl_example_generator = incontext_utils.BaselineIncontextExampleGenerator(incontext_settings,
                                                                                                    [positive_target])
                # Get subset of relevant of examples
                positive_selected_icl_examples = positive_icl_example_generator.select_icl_examples(
                    sys_kdma_name=kdma,
                    scenario_description_to_match=scenario_to_match,
                    prompt_to_match=prompt_to_match,
                    state_comparison=scenario_state
                )
                # Create positive ICL prompts
                for icl_sample in positive_selected_icl_examples:
                    positive_icl_examples.extend([
                        {"role": "user", "content": icl_sample['prompt']},
                        {"role": "assistant", "content": f'{icl_sample["response"]}'}
                    ])

                # Create negative ICL prompts
                if num_negative_samples > 0:
                    # Create negative ICL example generators
                    negative_target = {'kdma': kdma, 'name': name, 'value': negative_value}
                    negative_icl_example_generator = incontext_utils.BaselineIncontextExampleGenerator(incontext_settings,
                                                                                                    [negative_target])
                    # Get subset of relevant of examples
                    negative_selected_icl_examples = negative_icl_example_generator.select_icl_examples(
                        sys_kdma_name=kdma,
                        scenario_description_to_match=scenario_to_match,
                        prompt_to_match=prompt_to_match,
                        state_comparison=scenario_state
                    )
                    for icl_sample in negative_selected_icl_examples:
                        negative_icl_examples.extend([
                            {"role": "user", "content": icl_sample['prompt']},
                            {"role": "assistant", "content": f'{icl_sample["response"]}'}
                        ])
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

        action_to_take, dialog = self.populate_action_parameters(
            scenario_state,
            action_to_take,
            dialog)

        choice_info = {}
        return action_to_take, choice_info

    def populate_action_parameters(self, scenario_state, action_to_take, dialog):
        if action_to_take.action_type in {ActionTypeEnum.APPLY_TREATMENT,
                                          ActionTypeEnum.TAG_CHARACTER,
                                          ActionTypeEnum.CHECK_ALL_VITALS,
                                          ActionTypeEnum.CHECK_PULSE,
                                          ActionTypeEnum.CHECK_RESPIRATION,
                                          ActionTypeEnum.CHECK_BLOOD_OXYGEN,
                                          ActionTypeEnum.MOVE_TO_EVAC,
                                          ActionTypeEnum.MOVE_TO}:
            action_to_take, selected_character, selected_character_idx, dialog =\
                self.ensure_character_id_is_populated(scenario_state, action_to_take, dialog)

        if action_to_take.action_type == ActionTypeEnum.APPLY_TREATMENT:
            if action_to_take.parameters is None or 'treatment' not in action_to_take.parameters or 'location' not in action_to_take.parameters:
                # TODO: Add inference kwarg to use heurustic treatment options or not
                from align_system.algorithms.apply_treatment import treatment_options

                character_injuries = [i.to_dict() for i in scenario_state.characters[selected_character_idx].injuries]
                supplies = [s.to_dict() for s in scenario_state.supplies]

                heuristic_treatment_options = treatment_options(character_injuries, supplies)

                # Filter heuristic treatment options by already
                # populated treatment or location
                att_treatment = None
                att_location = None
                if action_to_take.parameters is not None:
                    att_treatment = action_to_take.parameters.get('treatment')
                    att_location = action_to_take.parameters.get('location')

                filtered_heuristic_treatments = []
                filtered_heuristic_params = []
                for heuristic_treatment, heuristic_params in zip(heuristic_treatment_options.get('treatments', ()),
                                                                 heuristic_treatment_options.get('parameters', ())):
                    if att_treatment is not None and heuristic_params['treatment'] != att_treatment:
                        continue
                    if att_location is not None and heuristic_params['location'] != att_location:
                        continue

                    filtered_heuristic_treatments.append(heuristic_treatment)
                    filtered_heuristic_params.append(heuristic_params)

                filtered_heuristic_treatment_options = copy.deepcopy(heuristic_treatment_options)
                filtered_heuristic_treatment_options['treatments'] = filtered_heuristic_treatments
                filtered_heuristic_treatment_options['parameters'] = filtered_heuristic_params

                # Should fall back to subsequent treatment / location
                # handler if no heuristic treatment options left after
                # filtering.
                if len(filtered_heuristic_treatments) > 0:
                    log.debug("[bold]*FILTERED HEURISTIC TREATMENT OPTIONS*[/bold]",
                              extra={"markup": True})
                    log.debug(filtered_heuristic_treatment_options)
                    action_to_take, selected_treatment, dialog =\
                        self.select_treatment_parameters(scenario_state,
                                                         action_to_take,
                                                         selected_character,
                                                         selected_character_idx,
                                                         dialog,
                                                         filtered_heuristic_treatment_options)
                else:
                    log.debug("[bold]*NO FILTERED HEURISTIC TREATMENT OPTIONS*[/bold]")

            # Use follow up prompt to define treatment and/or location if neccesary
            if action_to_take.parameters is None or 'treatment' not in action_to_take.parameters or 'location' not in action_to_take.parameters:
                action_to_take, selected_treatment, dialog =\
                    self.populate_treatment_parameters(scenario_state,
                                                       action_to_take,
                                                       selected_character,
                                                       selected_character_idx,
                                                       dialog)
        elif action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            if action_to_take.parameters is None or 'category' not in action_to_take.parameters:
                action_to_take, selected_tag, dialog =\
                    self.populate_tagging_parameters(scenario_state,
                                                     action_to_take,
                                                     selected_character,
                                                     selected_character_idx,
                                                     dialog)
        # Set aid_id for MOVE_TO_EVAC if missing
        elif action_to_take.action_type == ActionTypeEnum.MOVE_TO_EVAC:
            if action_to_take.parameters is None or "aid_id" not in action_to_take.parameters:
                action_to_take, selected_aid, dialog =\
                    self.populate_aid_parameters(scenario_state,
                                                 action_to_take,
                                                 selected_character,
                                                 selected_character_idx,
                                                 dialog)

        return action_to_take, dialog

    def ensure_character_id_is_populated(self,
                                         scenario_state,
                                         action_to_take,
                                         dialog):
        if action_to_take.character_id is None:
            # Use follow up prompt to define selected_character
            if action_to_take.action_type not in {ActionTypeEnum.MOVE_TO_EVAC,
                                                  ActionTypeEnum.MOVE_TO}:
                characters = [c for c in scenario_state.characters if not c.unseen]

            if action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
                # Further filtering for tagging action, don't tag
                # a character that already has a tag
                characters = [c for c in characters if c.tag is None]
            elif action_to_take.action_type in {ActionTypeEnum.CHECK_ALL_VITALS,
                                                ActionTypeEnum.CHECK_PULSE,
                                                ActionTypeEnum.CHECK_RESPIRATION,
                                                ActionTypeEnum.CHECK_BLOOD_OXYGEN}:
                # Further filtering for assessment actions, don't
                # allow an already "visited" character to be assessed
                # again; NOTE: Not certain this won't prevent us from
                # doing legitimate actions in some corner cases
                characters = [c for c in characters
                              if c.visited is None or not c.visited]

            dialog.append({'role': 'assistant',
                           'content': '{}  I would choose to {}'.format(
                               action_to_take.justification,
                               action_to_take.unstructured)})
            dialog.append({'role': 'user',
                           'content': followup_clarify_character(characters)})
            dialog_text = self.dialog_to_prompt(dialog)

            character_names = [c.name for c in characters]

            generator = outlines.generate.json(
                self.model,
                character_choice_json_schema(json.dumps(character_names)),
                sampler=self.sampler,
                whitespace_pattern=r"[ ]?")

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_character = generator(dialog_text)
            selected_character_idx = character_names.index(selected_character['character_choice'])

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_character, extra={"highlighter": JSON_HIGHLIGHTER})

            action_to_take.character_id = characters[selected_character_idx].id
        else:
            # Use action_to_take.character_id to define selected_character
            selected_character = {}
            for char_index in range(len(scenario_state.characters)):
                character = scenario_state.characters[char_index]
                if character.id == action_to_take.character_id:
                    selected_character['character_choice'] = character.name
                    selected_character_idx = char_index
                    break

            selected_character['brief_reasoning'] = action_to_take.justification

        return action_to_take, selected_character, selected_character_idx, dialog

    def populate_treatment_parameters(self,
                                      scenario_state,
                                      action_to_take,
                                      selected_character,
                                      selected_character_idx,
                                      dialog):
        # Get valid injury locations for the selected character
        valid_treatment_locations = []
        for injury in scenario_state.characters[selected_character_idx].injuries:
            valid_treatment_locations.append(injury.location)

        # Work-around when injuries are not yet discovered (intend actions)
        if len(valid_treatment_locations) == 0:
            log.info("No injuries on selected character. Allowing any treatment location")
            valid_treatment_locations = get_swagger_class_enum_values(InjuryLocationEnum)

        # If there is only one treatment location and we have the treatment, we don't need a follow-up
        if len(valid_treatment_locations) == 1 and action_to_take.parameters is not None and 'treatment' in action_to_take.parameters:
            action_to_take.parameters['location'] = valid_treatment_locations[0]

            selected_treatment = {'detailed_reasoning': '<Treatment already specified and only one valid treatment location>',
                                  'supplies_to_use': action_to_take.parameters['treatment'],
                                  'treatment_location': action_to_take.parameters['location']}
        # If there are multiple treatment locations and/or we are missing the treatment, use follow-up
        else:
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

            # Use follow-up response to define only the missing fields
            if action_to_take.parameters is None:
                action_to_take.parameters = {}
            if 'treatment' not in action_to_take.parameters:
                action_to_take.parameters['treatment'] = selected_treatment['supplies_to_use']
            if 'location' not in action_to_take.parameters:
                action_to_take.parameters['location'] = selected_treatment['treatment_location']

        return action_to_take, selected_treatment, dialog

    def select_treatment_parameters(self,
                                    scenario_state,
                                    action_to_take,
                                    selected_character,
                                    selected_character_idx,
                                    dialog,
                                    heuristic_treatment_options):
        possible_treatments = heuristic_treatment_options['treatments']

        # If there is only one treatment location and we have the
        # treatment, we don't need a follow-up
        if len(possible_treatments) == 0:
            #  TODO: Handle this case prior to calling this function
            raise RuntimeError("No possible treatments from heuristic_treatment_options!")
        elif len(possible_treatments) == 1:
            log.debug("[bold]*SELECTING ONLY REMAINING HEURISTIC TREATMENT OPTION*[/bold]")

            # Assumes correspondence between 'treatments' and 'parameters'
            assert len(heuristic_treatment_options['parameters']) == 1

            treatment_parameters = heuristic_treatment_options['parameters'][0]
            selected_treatment = {'detailed_reasoning': '<Only one heuristic treatment option available>',
                                  'supplies_to_use': treatment_parameters['treatment'],
                                  'treatment_location': treatment_parameters['location']}
        # If there are multiple treatment locations and/or we are missing the treatment, use follow-up
        else:
            available_supplies = [s for s in scenario_state.supplies if s.quantity > 0]

            dialog.append({'role': 'assistant',
                           'content': '{}  {} should receive the action.'.format(
                               selected_character['brief_reasoning'],
                               selected_character['character_choice'])})
            dialog.append({'role': 'user',
                           'content': followup_clarify_treatment_from_list(
                               scenario_state.characters[selected_character_idx],
                               available_supplies,
                               possible_treatments)})

            dialog_text = self.dialog_to_prompt(dialog)

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            generator = outlines.generate.json(
                self.model,
                treatment_choice_from_list_json_schema(
                    json.dumps(possible_treatments)),
                sampler=self.sampler,
                whitespace_pattern=r"[ ]?")

            selected_treatment = generator(dialog_text)
            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_treatment, extra={"highlighter": JSON_HIGHLIGHTER})

            treatment_idx = possible_treatments.index(selected_treatment['treatment_choice'])
            treatment_parameters = heuristic_treatment_options['parameters'][treatment_idx]

        # Use follow-up response to define only the missing fields
        if action_to_take.parameters is None:
            action_to_take.parameters = {}

        action_to_take.parameters = {**action_to_take.parameters, **treatment_parameters}

        return action_to_take, selected_treatment, dialog

    def populate_tagging_parameters(self,
                                    scenario_state,
                                    action_to_take,
                                    selected_character,
                                    selected_character_idx,
                                    dialog):
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

        return action_to_take, selected_tag, dialog

    def populate_aid_parameters(self,
                                scenario_state,
                                action_to_take,
                                selected_character,
                                selected_character_idx,
                                dialog):
        selected_character_dict =\
            scenario_state.characters[selected_character_idx].to_dict()

        # Limit to the aids that will accept the selected patient
        available_aids = [
            aid
            for aid in scenario_state.environment.decision_environment.aid
            if (
                aid.patients_treated is None or
                "military_disposition" not in selected_character_dict or
                selected_character_dict["miliary_disposition"] in aid.patients_treated
            )
        ]

        if len(available_aids) == 0:
            raise RuntimeError("No aids to choose from")
        elif len(available_aids) == 1:  # If there is only one option, we don't need a follow-up
            action_to_take.parameters["aid_id"] = available_aids[0].id

            selected_aid = {'brief_reasoning': '<Only one aid option available>',
                            'aid_choice': action_to_take.parameters["aid_id"]}
        else:
            dialog.append({'role': 'assistant',
                           'content': '{}  {} should receive the action.'.format(
                               selected_character['brief_reasoning'],
                               selected_character['character_choice'])})
            dialog.append({'role': 'user',
                           'content': followup_clarify_aid(
                                selected_character_dict,
                                available_aids)})

            dialog_text = self.dialog_to_prompt(dialog)

            generator = outlines.generate.json(
                self.model,
                aid_choice_json_schema(
                    json.dumps([aid.id for aid in available_aids])),
                sampler=self.sampler,
                whitespace_pattern=r"[ ]?")

            log.info("[bold]*DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

            selected_aid = generator(dialog_text)

            log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                     extra={"markup": True})
            log.info(selected_aid, extra={"highlighter": JSON_HIGHLIGHTER})

            action_to_take.parameters["aid_id"] = selected_aid['aid_choice']

        return action_to_take, selected_aid, dialog
