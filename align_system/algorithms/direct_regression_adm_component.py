import copy
import inspect
import re
from collections.abc import Mapping

from rich.highlighter import JSONHighlighter
import ubelt as ub
from jinja2 import Template

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class DefaultSchemaTemplate:
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
      "minimum": 0,
      "maximum": 100
    }
  },
  "required": ["reasoning", "score"],
  "additionalProperties": false
}'''


class DirectRegressionADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine,
                 per_attribute_templates,
                 num_samples=1,
                 target_attribute_names_override=None,
                 enable_caching=False):
        self.structured_inference_engine = structured_inference_engine

        self.per_attribute_templates = per_attribute_templates

        self.num_samples = num_samples

        self.target_attribute_names_override = target_attribute_names_override
        self.enable_caching = enable_caching

    def run_returns(self):
        return ('attribute_prediction_reasonings',
                'attribute_prediction_scores')

    def run(self,
            scenario_state,
            choices,
            actions,
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

        if self.enable_caching:
            scenario_state_copy = copy.deepcopy(scenario_state)
            if hasattr(scenario_state, 'elapsed_time'):
                # Don't consider the elapsed_time of the state when caching
                scenario_state_copy.elapsed_time = 0

            depends = '\n'.join((
                self.cache_repr(),
                repr(scenario_state_copy),
                repr(choices),
                repr(actions),
                repr(icl_dialog_elements),
                repr(target_attribute_names)))

            cacher = ub.Cacher('direct_regression_adm_component', depends, verbose=0)
            log.debug(f'cacher.fpath={cacher.fpath}')

            cached_output = cacher.tryload()
            if cached_output is not None:
                log.info("Cache hit for `direct_regression_adm_component`"
                         " returning cached output")
                return cached_output
            else:
                log.info("Cache miss for `direct_regression_adm_component` ..")

        # Convert input datastructures into dicts if not already
        actions = [a.to_dict() for a in actions if not isinstance(a, Mapping)]
        if not isinstance(scenario_state, Mapping):
            scenario_state = scenario_state.to_dict()

        attribute_prediction_scores = {}
        attribute_prediction_reasonings = {}
        for attribute in target_attribute_names:
            if attribute not in self.per_attribute_templates:
                raise RuntimeError(f"Missing {attribute} from self.per_attribute_templates")

            missing_character_default_value = self.per_attribute_templates[attribute].get(
                'missing_character_default_value', 0)
            for choice, action in zip(choices, actions):
                if 'character_id' not in action:
                    attribute_prediction_scores.setdefault(choice, {})[attribute] =\
                        [missing_character_default_value] * self.num_samples
                    attribute_prediction_reasonings.setdefault(choice, {})[attribute] =\
                        "<No character_id specified for action>"

                    log.info(f"No character_id specified for action ('{choice}'); "
                             f"assigning {missing_character_default_value} for {attribute}")

                    continue

                dialog = []
                if self.per_attribute_templates[attribute].get('system_prompt') is not None:
                    dialog.insert(0, DialogElement(
                        role='system',
                        content=self.per_attribute_templates[attribute]['system_prompt'],
                        tags=['regression']))

                selected_character = None
                for character in scenario_state['characters']:
                    if character['id'] == action['character_id']:
                        selected_character = character
                        break

                if selected_character is None:
                    raise RuntimeError(f"Couldn't find character ({action['character_id']}) referenced by action")

                prompt_template = self.per_attribute_templates[attribute]['prompt_template']
                if callable(prompt_template):
                    prompt = call_with_coerced_args(
                        prompt_template,
                        {'character': selected_character})
                elif isinstance(prompt_template, str):
                    prompt = Template(prompt_template).render(
                        {'character': selected_character})
                else:
                    raise RuntimeError("prompt_template is of an unexpected type")

                dialog.append(DialogElement(role='user',
                                            content=prompt,
                                            tags=['regression']))

                dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

                log.info(f"[bold]*{attribute.upper()} PREDICTION DIALOG PROMPT*[/bold]",
                         extra={"markup": True})
                log.info(dialog_prompt)

                output_schema = call_with_coerced_args(
                    self.per_attribute_templates[attribute]['schema_template'], {})

                responses = self.structured_inference_engine.run_inference(
                    [dialog_prompt] * self.num_samples, output_schema)

                for i, response in enumerate(responses):
                    log.info(f"[bold]*{attribute.upper()} PREDICTION RESPONSE (sample #{i})*[/bold]", extra={"markup": True})
                    log.info(response, extra={"highlighter": JSON_HIGHLIGHTER})

                factor = self.per_attribute_templates[attribute].get('factor', 100)
                attribute_prediction_scores.setdefault(choice, {})[attribute] =\
                    [r['score'] / float(factor) for r in responses]
                attribute_prediction_reasonings.setdefault(choice, {})[attribute] =\
                    [r['reasoning'] for r in responses]

        outputs = (attribute_prediction_reasonings,
                   attribute_prediction_scores)

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

        per_attribute_templates_repr = {}
        for attribute, templates in self.per_attribute_templates.items():
            for template_name, template in templates.items():
                if isinstance(template, str) or isinstance(template, int):
                    template_repr = template
                else:
                    template_repr = _generic_object_repr(template)

            per_attribute_templates_repr.setdefault(attribute, {})[template_name] = template_repr

        return re.sub(r'^\s+', '',
                      f"""
                       {self.__class__.__module__}.{self.__class__.__name__}(
                       structured_inference_engine={self.structured_inference_engine.cache_repr()},
                       per_attribute_templates={per_attribute_templates_repr},
                       num_samples={self.num_samples},
                       target_attribute_names_override={self.target_attribute_names_override},
                       )""", flags=re.MULTILINE).strip()
