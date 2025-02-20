import copy

from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.utils.outlines_prompts_utils import (
    get_unique_structured_character_info,
    get_relevant_structured_character_info)
from align_system.prompt_engineering.outlines_prompts import (
    comparative_kdma_score_prediction_prompt,
    enum_comparative_kdma_score_prediction_json_schema,
    comparative_kdma_score_prediction_json_schema,
    scenario_state_description_with_relevant_char_info)
from align_system.data_models.dialog import DialogElement


class ComparativeRegressionADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine,
                 attributes={},
                 num_samples=1,
                 enum_scores=False):
        self.structured_inference_engine = structured_inference_engine

        self.attributes = attributes

        self.num_samples = num_samples
        self.enum_scores = enum_scores

    def _build_scenario_description(self,
                                    scenario_state,
                                    target_attributes):
        relevant_fields = []
        for attribute in target_attributes:
            relevant_fields.extend(attribute.relevant_structured_character_info)

        if 'all_unique' in relevant_fields:
            character_info = get_unique_structured_character_info(scenario_state.characters)
        else:
            character_info = get_relevant_structured_character_info(
                scenario_state.characters,
                [dict(v) for v in self.attributes.values()])

        scenario_description = scenario_state_description_with_relevant_char_info(scenario_state, character_info)

        return scenario_description

    def run(self,
            scenario_state,
            choice_evaluation,
            dialogs,
            alignment_target=None):
        # Assuming we only have a single dialog coming in at this
        # point.  Perhaps the code / architecture should be more
        # robust to the fanning out of dialogs
        if len(dialogs) != 1:
            raise RuntimeError("Assumption violated: only expecting a single "
                               "dialog at this point")

        if alignment_target is None:
            target_attribute_names = []
        else:
            target_attribute_names = attributes_in_alignment_target(alignment_target)

        target_attributes = [self.attributes[n] for n in target_attribute_names]

        scenario_description = self._build_scenario_description(
            scenario_state, target_attributes)

        choices = list(choice_evaluation.keys())

        # Pull in predicted outcomes from `cohice_evaluation` if populated
        outcome_predictions =\
            {k: {'predicted_outcome': v.get('predicted_outcomes', None)}
             for k, v in choice_evaluation.items()}

        output_dialogs = []
        for attribute in target_attributes:
            attribute_dialogs = []
            for _dialog in dialogs:
                for sample_idx in range(self.num_samples):
                    # Want to make a copy as we may modify the same dialog
                    # more than once due to samples and/or multiple-kdmas
                    dialog = copy.deepcopy(_dialog)

                    predict_kdma_prompt = comparative_kdma_score_prediction_prompt(
                        scenario_description,
                        outcome_predictions,
                        attribute.name)

                    dialog.append(DialogElement(role='user',
                                                content=predict_kdma_prompt,
                                                namespace='.',
                                                tags=['regression']))

                    attribute_dialogs.append(dialog)

            if self.enum_scores:
                score_schema = enum_comparative_kdma_score_prediction_json_schema(
                    choices, attribute.valid_scores)
            else:
                score_schema = comparative_kdma_score_prediction_json_schema(
                    choices, attribute.factor)

            dialog_prompts = [self.structured_inference_engine.dialog_to_prompt(d)
                              for d in attribute_dialogs]

            attribute_score_responses =\
                self.structured_inference_engine.run_inference(
                    dialog_prompts, score_schema)

            for dialog, response in zip(attribute_dialogs, attribute_score_responses):
                dialog.append(DialogElement(role='assistant',
                                            content=str(response),
                                            namespace='.',
                                            tags=['regression']))

            # Adds a copy of the dialog for each sample; could
            # consider only adding a single dialog since they're
            # duplicated for each sample (at least with current
            # implementation)
            output_dialogs.extend(attribute_dialogs)

            for response in attribute_score_responses:
                for choice, choice_eval in choice_evaluation.items():
                    reasonings = choice_eval.setdefault('kdma_prediction_reasonings', {})
                    scores = choice_eval.setdefault('kdma_prediction_scores', {})

                    reasonings.setdefault(attribute.kdma, []).append(
                        response[choice]['reasoning'])
                    scores.setdefault(attribute.kdma, []).append(
                        response[choice]['score'] / attribute.factor)

        return choice_evaluation, output_dialogs
