from rich.highlighter import JSONHighlighter

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement
from align_system.utils.voting import calculate_votes

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class OutlinesBaselineADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine,
                 scenario_description_template,
                 prompt_template,
                 output_schema_template,
                 system_prompt_template=None,
                 num_samples=1,
                 vote_calculator_fn=calculate_votes):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template
        self.output_schema_template = output_schema_template

        self.system_prompt_template = system_prompt_template

        self.num_samples = num_samples
        self.vote_calculator_fn = vote_calculator_fn

    def run_returns(self):
        return ('chosen_choice', 'justification', 'dialog')

    def run(self,
            scenario_state,
            choices):
        scenario_description = call_with_coerced_args(
            self.scenario_description_template,
            {'scenario_state': scenario_state})

        dialog = []
        if self.system_prompt_template is not None:
            system_prompt = call_with_coerced_args(
                self.system_prompt_template, {})

            dialog.insert(0, DialogElement(role='system',
                                           content=system_prompt,
                                           tags=['regression']))

        prompt = call_with_coerced_args(
            self.prompt_template,
            {'scenario_state': scenario_state,
             'scenario_description': scenario_description,
             'choices': choices})

        dialog.append(DialogElement(role='user',
                                    content=prompt,
                                    tags=['regression']))

        output_schema = call_with_coerced_args(
            self.output_schema_template,
            {'choices': choices})

        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

        log.info("[bold]*KDMA SCORE PREDICTION DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(dialog_prompt)

        responses = self.structured_inference_engine.run_inference(
            [dialog_prompt] * self.num_samples, output_schema)

        votes = self.vote_calculator_fn(
            choices, [r['action_choice'] for r in responses])

        log.explain("[bold]*VOTES*[/bold]",
                    extra={"markup": True})
        log.explain(votes, extra={"highlighter": JSON_HIGHLIGHTER})

        # Take top choice by score (votes is a dictionary of choice: score)
        top_choice, top_choice_score = max(votes.items(), key=lambda x: x[1])

        # Grab justification for top_choice (just taking first
        # instance we find)
        top_choice_justification = ""
        for response in responses:
            if response['action_choice'] == top_choice:
                top_choice_justification = response['detailed_reasoning']
                break

        return top_choice, top_choice_justification, dialog
