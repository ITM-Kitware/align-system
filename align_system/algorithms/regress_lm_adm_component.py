import re
import inspect
import copy

from regress_lm import core
from regress_lm import rlm
from regress_lm.models.pytorch import t5gemma_model

from rich.highlighter import JSONHighlighter
import ubelt as ub

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class RegressLMADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine,
                 scenario_description_template,
                 prompt_template,
                 score_schema_template,
                 attributes=None,
                 system_prompt_template=None,
                 num_samples=1,
                 enum_scores=False,
                 target_attribute_names_override=None,
                 enable_caching=False):
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

        self.target_attribute_names_override = target_attribute_names_override

        self.enable_caching = enable_caching
        # self.reg_lm = rlm.RegressLM.from_default(max_input_len=2048, learning_rate=1e-3)
        
        model = t5gemma_model.T5GemmaModel('google/t5gemma-s-s-prefixlm')
        self.reg_lm = rlm.RegressLM.from_default(max_input_len=2048, learning_rate=1e-3, model=model)

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

        if self.target_attribute_names_override is not None:
            overridden_target_attribute_names = []
            for attribute_name in self.target_attribute_names_override:
                if attribute_name == '*':
                    # '*' in the override means to include the attribute names
                    # from the target (in addition to whatever else is
                    # specified in the override)
                    overridden_target_attribute_names.extend(target_attribute_names)
                else:
                    overridden_target_attribute_names.append(attribute_name)

            target_attribute_names = overridden_target_attribute_names

        target_attributes = [self.attributes[n] for n in target_attribute_names]

        attribute_dialogs = {}
        attribute_prediction_scores = {}
        attribute_prediction_reasonings = {}
        for attribute in target_attributes:

            examples = []
            for i in range(0, len(icl_dialog_elements[attribute.kdma]), 2):
                icl_prompt = icl_dialog_elements[attribute.kdma][i].content
                icl_response = icl_dialog_elements[attribute.kdma][i+1].content
                examples.append(core.Example(x=icl_prompt, y=float(icl_response)))
            self.reg_lm.fine_tune(examples, examples)

            scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {'scenario_state': scenario_state,
                 'alignment_target': alignment_target,
                 'attribute': attribute.name,
                 'attributes_of_interest': {attribute.name,}})

            for choice in choices:
                predict_kdma_prompt = call_with_coerced_args(
                    self.prompt_template,
                    {'scenario_description': scenario_description,
                    'choice': choice,
                    'attribute': attribute.name})

                query = core.ExampleInput(x=predict_kdma_prompt)
                predicted_scores = self.reg_lm.sample([query], num_samples=30)[0]
                predicted_score = float(sum(predicted_scores)/len(predicted_scores))
                print(predicted_score)
                
                attribute_prediction_scores.setdefault(choice, {})
                attribute_prediction_scores[choice].setdefault(
                        attribute.kdma, []).append(predicted_score / attribute.factor)

                attribute_prediction_reasonings.setdefault(choice, {})
                attribute_prediction_reasonings[choice].setdefault(
                            attribute.kdma, []).append('Regress-LM prediction.')
            

            attribute_dialogs[attribute.kdma] = ''

        outputs = (attribute_prediction_reasonings, attribute_prediction_scores, attribute_dialogs)
        return outputs