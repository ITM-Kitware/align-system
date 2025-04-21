from rich.highlighter import JSONHighlighter

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class ComparativeRegressionADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine,
                 scenario_description_template,
                 prompt_template,
                 score_schema_template,
                 attributes=None,
                 system_prompt_template=None,
                 num_samples=1,
                 enum_scores=False):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template
        self.score_schema_template = score_schema_template

        if attributes is None:
            attributes = {}
        self.attributes = attributes

        self.system_prompt_template = system_prompt_template

        self.num_samples = num_samples
        self.enum_scores = enum_scores

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
            scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {'scenario_state': scenario_state,
                 'alignment_target': alignment_target,
                 'attribute': attribute.name,
                 'attributes_of_interest': {attribute.name,}})

            dialog = []
            if self.system_prompt_template is not None:
                system_prompt = call_with_coerced_args(
                    self.system_prompt_template,
                    {'target_attribute': attribute})

                dialog.insert(0, DialogElement(role='system',
                                               content=system_prompt,
                                               namespace='.',
                                               tags=['regression']))

            # If we get icl_dialog_elements, include them in the
            # dialog, maybe a more explicit argument (wether or not to
            # use icl) makes more sense?
            if len(icl_dialog_elements) > 0:
                dialog.extend(icl_dialog_elements)

            predict_kdma_prompt = call_with_coerced_args(
                self.prompt_template,
                {'scenario_state': scenario_state,
                 'scenario_description': scenario_description,
                 'choices': choices,
                 'choice_outcomes': {c: None for c in choices},
                 'attribute': attribute.name})

            dialog.append(DialogElement(role='user',
                                        content=predict_kdma_prompt,
                                        namespace='.',
                                        tags=['regression']))

            score_schema = call_with_coerced_args(
                self.score_schema_template,
                {'choices': choices,
                 'attribute': attribute.name})

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
                    attribute_prediction_scores.setdefault(choice, {})
                    attribute_prediction_scores[choice].setdefault(
                        attribute.kdma, []).append(response[choice]['score'] / attribute.factor)

                    attribute_prediction_reasonings.setdefault(choice, {})
                    attribute_prediction_reasonings[choice].setdefault(
                        attribute.kdma, []).append(response[choice]['reasoning'])

            attribute_dialogs[attribute.kdma] = dialog

        return attribute_prediction_reasonings, attribute_prediction_scores, attribute_dialogs
