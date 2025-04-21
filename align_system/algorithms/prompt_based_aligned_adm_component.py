from rich.highlighter import JSONHighlighter
from swagger_client.models import KDMAValue

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement
from align_system.utils.voting import (
    calculate_votes,
    filter_votes_to_responses,
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class PromptBasedAlignedADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine,
                 scenario_description_template,
                 prompt_template,
                 output_schema_template,
                 system_prompt_template,
                 num_positive_samples=1,
                 num_negative_samples=0,
                 vote_calculator_fn=calculate_votes,
                 filter_votes_to_positives=True,
                 shuffle_choices=True):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template
        self.output_schema_template = output_schema_template

        self.system_prompt_template = system_prompt_template

        self.num_positive_samples = num_positive_samples
        self.num_negative_samples = num_negative_samples

        self.vote_calculator_fn = vote_calculator_fn
        self.filter_votes_to_positives = filter_votes_to_positives

        self.shuffle_choices = shuffle_choices

    def run_returns(self):
        return ('chosen_choice', 'justification', 'dialog')

    def run(self,
            scenario_state,
            choices,
            alignment_target,
            positive_icl_dialog_elements=[],
            negative_icl_dialog_elements=[]):
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

        scenario_description = call_with_coerced_args(
            self.scenario_description_template,
            {'scenario_state': scenario_state})

        prompt = call_with_coerced_args(
            self.prompt_template,
            {'scenario_state': scenario_state,
             'scenario_description': scenario_description,
             'choices': choices})

        positive_dialog = []
        if self.system_prompt_template is not None:
            positive_system_prompt = call_with_coerced_args(
                self.system_prompt_template,
                {'target_kdma': kdma,
                 'target_value': value})

            positive_dialog.insert(
                0, DialogElement(role='system',
                                 content=positive_system_prompt,
                                 tags=['regression']))

        if len(positive_icl_dialog_elements) > 0:
            positive_dialog.extend(positive_icl_dialog_elements)

        positive_dialog.append(
            DialogElement(role='user',
                          content=prompt,
                          tags=['regression']))

        positive_dialog_prompt = self.structured_inference_engine.dialog_to_prompt(
            positive_dialog)

        log.info("[bold]*POSITIVE DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(positive_dialog_prompt)

        if self.num_negative_samples > 0:
            negative_dialog = []
            if self.system_prompt_template is not None:
                negative_system_prompt = call_with_coerced_args(
                    self.system_prompt_template,
                    {'target_kdma': kdma,
                     'target_value': negative_value})

                negative_dialog.insert(
                    0, DialogElement(role='system',
                                     content=negative_system_prompt,
                                     tags=['regression']))

            if len(negative_icl_dialog_elements) > 0:
                negative_dialog.extend(negative_icl_dialog_elements)

            negative_dialog.append(
                DialogElement(role='user',
                              content=prompt,
                              tags=['regression']))

            negative_dialog_prompt = self.structured_inference_engine.dialog_to_prompt(
                negative_dialog)

            log.info("[bold]*NEGATIVE DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(negative_dialog_prompt)

        output_schema = call_with_coerced_args(
            self.output_schema_template,
            {'choices': choices})

        positive_responses = self.structured_inference_engine.run_inference(
            [positive_dialog_prompt] * self.num_positive_samples, output_schema)
        positive_choices = [r['action_choice'] for r in positive_responses]
        for i, positive_response in enumerate(positive_responses):
            log.info("[bold]*POSITIVE RESPONSE ({}, sample #{})*[/bold]".format(
                     kdma, i), extra={"markup": True})
            log.info(positive_response, extra={"highlighter": JSON_HIGHLIGHTER})

        if self.num_negative_samples > 0:
            negative_responses = self.structured_inference_engine.run_inference(
                [negative_dialog_prompt] * self.num_negative_samples, output_schema)
            negative_choices = [r['action_choice'] for r in negative_responses]
            for i, negative_response in enumerate(negative_responses):
                log.info("[bold]*NEGATIVE RESPONSE ({}, sample #{})*[/bold]".format(
                         kdma, i), extra={"markup": True})
                log.info(negative_response, extra={"highlighter": JSON_HIGHLIGHTER})

        else:
            negative_choices = None

        votes = self.vote_calculator_fn(
            choices, positive_choices, negative_choices)

        log.explain("[bold]*VOTES*[/bold]",
                    extra={"markup": True})
        log.explain(votes, extra={"highlighter": JSON_HIGHLIGHTER})

        if self.filter_votes_to_positives:
            filtered_votes = filter_votes_to_responses(
                votes, positive_choices)

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
        for response in positive_responses:
            if response['action_choice'] == top_choice:
                top_choice_justification = response['detailed_reasoning']
                break

        return top_choice, top_choice_justification, positive_dialog
