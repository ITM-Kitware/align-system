import re
import inspect
import copy
from collections.abc import Mapping

from rich.highlighter import JSONHighlighter
import ubelt as ub

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class DefaultRegressionTemplate:
    def __call__(self, character):
        return f"""Given the injury described below, provide a reasoning and a medical urgency rating:

{character["unstructured"]}
"""

class DefaultMedicalUrgencySchemaTemplate:
    def __call__(self):
        return '''
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "maxLength": 512
    },
    "score": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100
    }
  },
  "required": ["reasoning", "score"],
  "additionalProperties": false
}'''


DEFAULT_SYSTEM_PROMPT = """Here's a list of injuries and they're associated "medical urgency" scores, where a 100 indicates that immediate medical attention is needed in a triage situation; and a 1 indicates the lowest priority:

Massive bleeding from the abdomen or torso would score a 100.
Massive bleeding from the head or neck would score a 99.
Massive bleeding from the hip, groin, or shoulder would score a 99.
Massive bleeding from a limb would score 98.

Not breathing would score 95.
Gasping for breath with a chest wound would score 94.
Not breathing after airway repositioned would score 92.

Moderate bleeding from the abdomen or torso would score a 60.
Moderate bleeding from the hip, going, or shoulder would score a 59.

Signs of decompensated shock would score 60.

Severe burns over 50% of the body would score 64.
Severe burns over 20% of the body would score 53.

A compound fracture with mild bleeding would score 41.
A closed fracture with no bleeding would score a 1.
A dislocation with no bleeding would score a 1.

You'll be given an injury description and asked to assign a "medical urgency" score between 1 to 100 along with a reasoning for your score.  Ignore any details that are not relevant to the injury.  Your response should be formatted as valid JSON.
"""


class DirectMedicalRegressionADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine,
                 character_injury_description_template=DefaultRegressionTemplate(),
                 medical_urgency_schema_template=DefaultMedicalUrgencySchemaTemplate(),
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 num_samples=1,
                 enable_caching=False):
        self.structured_inference_engine = structured_inference_engine

        self.character_injury_description_template = character_injury_description_template
        self.medical_urgency_schema_template = medical_urgency_schema_template
        self.system_prompt = system_prompt

        self.num_samples = num_samples

        self.enable_caching = enable_caching


    def run_returns(self):
        return ('medical_prediction_reasonings',
                'attribute_prediction_scores')

    def run(self,
            scenario_state,
            choices,
            actions):
        if self.enable_caching:
            scenario_state_copy = copy.deepcopy(scenario_state)
            if hasattr(scenario_state, 'elapsed_time'):
                # Don't consider the elapsed_time of the state when caching
                scenario_state_copy.elapsed_time = 0

            depends = '\n'.join((
                self.cache_repr(),
                repr(scenario_state_copy),
                repr(actions)))

            cacher = ub.Cacher('direct_medical_regression_adm_component', depends, verbose=0)
            log.debug(f'cacher.fpath={cacher.fpath}')

            cached_output = cacher.tryload()
            if cached_output is not None:
                log.info("Cache hit for `direct_medical_regression_adm_component`"
                         " returning cached output")
                return cached_output
            else:
                log.info("Cache miss for `direct_medical_regression_adm_component` ..")

        # Convert input datastructures into dicts if not already
        actions = [a.to_dict() for a in actions if not isinstance(a, Mapping)]
        if not isinstance(scenario_state, Mapping):
            scenario_state = scenario_state.to_dict()

        attribute_prediction_scores = {}
        attribute_prediction_reasonings = {}

        for choice, action in zip(choices, actions):
            if 'character_id' not in action:
                attribute_prediction_scores.setdefault(choice, {})['medical'] = [0.0] * self.num_samples
                attribute_prediction_reasonings.setdefault(choice, {})['medical'] =\
                    "<No character_id specified for action>"

                log.info(f"No character_id specified for action ('{choice}'); assigning 0.0 for medical urgency")

                continue

            dialog = []
            if self.system_prompt is not None:
                dialog.insert(0, DialogElement(role='system',
                                               content=self.system_prompt,
                                               tags=['regression']))

            selected_character = None
            for character in scenario_state['characters']:
                if character['id'] == action['character_id']:
                    selected_character = character
                    break

            if selected_character is None:
                raise RuntimeError(f"Couldn't find character ({action['character_id']}) referenced by action")

            predict_medical_urgency_prompt = call_with_coerced_args(
                self.character_injury_description_template,
                {'character': selected_character})

            dialog.append(DialogElement(role='user',
                                        content=predict_medical_urgency_prompt,
                                        tags=['regression']))

            medical_urgency_schema = call_with_coerced_args(
                self.medical_urgency_schema_template, {})

            dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

            log.info("[bold]*MEDICAL URGENCY PREDICTION DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_prompt)

            responses = self.structured_inference_engine.run_inference(
                    [dialog_prompt] * self.num_samples, medical_urgency_schema)

            for i, response in enumerate(responses):
                log.info("[bold]*MEDICAL URGENCY PREDICTION RESPONSE (sample #{})*[/bold]".format(i), extra={"markup": True})
                log.info(response, extra={"highlighter": JSON_HIGHLIGHTER})

            # ** IMPORTANT ** Dividing by hardcoded factor of 100.0 here
            attribute_prediction_scores.setdefault(choice, {})['medical'] =\
                [r['score'] / 100.0 for r in responses]
            attribute_prediction_reasonings.setdefault(choice, {})['medical'] =\
                [r['reasoning'] for r in responses]

        outputs = (attribute_prediction_reasonings, attribute_prediction_scores)

        if self.enable_caching:
            cacher.save(outputs)

        return outputs

    def cache_repr(self):
        '''
        Return a string representation of this object for caching;
        .i.e. if the return value of this function is the same for two
        object instances, it's assumed that `run` output will be
        the same if given the same parameters
        '''

        def _generic_object_repr(obj):
            init_params = inspect.signature(obj.__class__.__init__).parameters
            obj_vars = vars(obj)

            return "{}.{}({})".format(
                obj.__class__.__module__,
                obj.__class__.__name__,
                ", ".join([f"{p}={obj_vars[p]}" for p in init_params
                           if p != 'self' and p != 'args' and p != 'kwargs']))

        return re.sub(r'^\s+', '',
                      f"""
                       {self.__class__.__module__}.{self.__class__.__name__}(
                       structured_inference_engine={self.structured_inference_engine.cache_repr()},
                       character_injury_description_template={_generic_object_repr(self.character_injury_description_template)},
                       medical_urgency_schema_template={_generic_object_repr(self.medical_urgency_schema_template)},
                       system_prompt={self.system_prompt},
                       num_samples={self.num_samples},
                       )""", flags=re.MULTILINE).strip()
