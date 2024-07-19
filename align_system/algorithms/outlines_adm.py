import json
import random
import itertools

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
from align_system.utils.voting import calculate_votes
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

    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_positive_samples=1,
                                num_negative_samples=0,
                                shuffle_choices=False,
                                **kwargs):
        if self.baseline and num_negative_samples > 0:
            raise RuntimeError("No notion of negative samples for baseline run")

        scenario_description = scenario_state_description_1(scenario_state)
        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = [a.unstructured for a in available_actions]

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
        else:
            positive_system_prompt = baseline_system_prompt()
            if num_negative_samples > 0:
                raise RuntimeError("No notion of negative samples for baseline run")

        positive_dialogs = []
        for _ in range(num_positive_samples):
            shuffled_choices = random.sample(choices, len(choices))

            prompt = action_selection_prompt(scenario_description, shuffled_choices)
            dialog = [{'role': 'system', 'content': positive_system_prompt},
                      {'role': 'user', 'content': prompt}]

            positive_dialogs.append(dialog)

        negative_dialogs = []
        for _ in range(num_negative_samples):
            shuffled_choices = random.sample(choices, len(choices))

            prompt = action_selection_prompt(scenario_description, shuffled_choices)
            dialog = [{'role': 'system', 'content': negative_system_prompt},
                      {'role': 'user', 'content': prompt}]

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

        responses = generator(dialog_texts)

        if len(dialog_texts) == 1:
            # Ensure responses is a list in the case that we passed a
            # single dialog text
            responses = [responses]

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
