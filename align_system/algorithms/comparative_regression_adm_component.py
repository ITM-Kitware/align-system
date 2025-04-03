import copy

from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.utils.outlines_prompts_utils import (
    get_unique_structured_character_info,
    get_relevant_structured_character_info)
from align_system.prompt_engineering.outlines_prompts import (
    comparative_kdma_score_prediction_prompt,
    enum_comparative_kdma_score_prediction_json_schema,
    comparative_kdma_score_prediction_json_schema,
    comparative_kdma_score_prediction_system_prompt,
    comparative_kdma_score_prediction_system_prompt_with_examples,
    scenario_state_description_with_relevant_char_info)
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class ComparativeRegressionADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine,
                 scenario_description_template,
                 prompt_template,
                 score_schema_template,
                 attributes={},
                 inject_system_prompt=True,
                 score_examples_in_system_prompt=False,
                 num_samples=1,
                 enum_scores=False):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template
        self.score_schema_template = score_schema_template

        self.attributes = attributes

        self.inject_system_prompt = inject_system_prompt
        self.score_examples_in_system_prompt = score_examples_in_system_prompt

        self.num_samples = num_samples
        self.enum_scores = enum_scores

    def _build_system_prompt(self, target_attribute):
        if self.score_examples_in_system_prompt:
            template = self.environment.from_string(
                target_attribute.score_examples)
            score_examples = template.render(
                kdma_scale_factor=target_attribute.factor)
            kdma_score_sys_prompt = comparative_kdma_score_prediction_system_prompt_with_examples(
                target_attribute.name,
                target_attribute.description,
                score_examples,
                target_attribute.factor)
        else:
            kdma_score_sys_prompt = comparative_kdma_score_prediction_system_prompt(
                target_attribute.name,
                target_attribute.description,
                target_attribute.factor)

        return kdma_score_sys_prompt

    def run_returns(self):
        return ('attribute_prediction_reasonings',
                'attribute_prediction_scores',
                'attribute_dialogs')

    def run(self,
            scenario_state,
            choices,
            icl_dialog_elements=[],
            alignment_target=None):
        if alignment_target is None:
            target_attribute_names = []
        else:
            target_attribute_names = attributes_in_alignment_target(alignment_target)

        target_attributes = [self.attributes[n] for n in target_attribute_names]

        attribute_dialogs = {}
        attribute_prediction_scores = {}
        attribute_prediction_reasonings = {}
        for attribute in target_attributes:
            scenario_description = self.scenario_description_template(
                scenario_state, alignment_target, {attribute.name,})

            dialog = []
            if self.inject_system_prompt:
                system_prompt = self._build_system_prompt(attribute)

            dialog.insert(0, DialogElement(role='system',
                                           content=system_prompt,
                                           namespace='.',
                                           tags=['regression']))

            if len(icl_dialog_elements) > 0:
                dialog.extend(icl_dialog_elements)

            predict_kdma_prompt = self.prompt_template(
                scenario_state,
                scenario_description,
                choices,
                {attribute.name,})

            dialog.append(DialogElement(role='user',
                                        content=predict_kdma_prompt,
                                        namespace='.',
                                        tags=['regression']))

            for sample_idx in range(self.num_samples):
                attribute_dialogs.append(dialog)

            score_schema = self.score_schema_template(
                choices, {attribute.name,})

            dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

            log.info("[bold]*KDMA SCORE PREDICTION DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_prompt)

            responses = self.structured_inference_engine.run_inference(
                    [dialog_prompt] * self.num_samples, score_schema)

            for i, response in enumerate(responses):
                log.info("[bold]*KDMA SCORE PREDICTION RESPONSE ({}, sample #{})*[/bold]".format(
                    attribute.kdma, i), extra={"markup": True})
                log.info(response, extra={"highlighter": JSON_HIGHLIGHTER})

                for choice in choices:
                    attribute_prediction_scores[choice].setdefault(
                        attribute.kdma, []).append(response[choice]['score'] / attribute.factor)

                    attribute_prediction_reasonings[choice].setdefault(
                        attribute.kdma, []).append(response[choice]['reasoning'])

            attribute_dialogs[attribute.kdma] = dialog

        return attribute_prediction_reasonings, attribute_prediction_scores, attribute_dialogs
