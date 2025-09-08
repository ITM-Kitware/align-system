import re
import inspect
import copy

from rich.highlighter import JSONHighlighter
import ubelt as ub

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class RegressionOutputsConflictResolver:
    def __call__(self, output_name, a, b):
        if (output_name == 'attribute_prediction_reasonings'
            or output_name == 'attribute_prediction_scores'):
            # Append regression values to existing outputs if there's
            # a conflict
            output_dict = copy.deepcopy(a)

            for choice, attr_pred in b.items():
                for attr, pred in attr_pred.items():
                    output_dict.setdefault(choice, {})
                    output_dict[choice].setdefault(attr, []).extend(pred)

            return output_dict
        else:
            # No special handling of other regression outputs like
            # `attribute_dialogs`, could add if needed
            return b

class ComparativeRegressionADMComponent(ADMComponent):
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
                 enable_caching=False,
                 reverse_choice_ordering=False,
                 output_conflict_resolver=None):
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

        self.reverse_choice_ordering = reverse_choice_ordering

        self.output_conflict_resolver = output_conflict_resolver

    def run_returns(self):
        return ('attribute_prediction_reasonings',
                'attribute_prediction_scores',
                'attribute_dialogs')

    def run(self,
            scenario_state,
            choices,
            icl_dialog_elements=[],
            alignment_target=None):
        if self.reverse_choice_ordering:
            choices = list(reversed(choices))

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

        if self.enable_caching:
            scenario_state_copy = copy.deepcopy(scenario_state)
            if hasattr(scenario_state, 'elapsed_time'):
                # Don't consider the elapsed_time of the state when caching
                scenario_state_copy.elapsed_time = 0

            depends = '\n'.join((
                self.cache_repr(),
                repr(scenario_state_copy),
                repr(choices),
                repr(icl_dialog_elements),
                repr(target_attribute_names)))

            cacher = ub.Cacher('comparative_regression_adm_component', depends, verbose=0)
            log.debug(f'cacher.fpath={cacher.fpath}')

            cached_output = cacher.tryload()
            if cached_output is not None:
                log.info("Cache hit for `comparative_regression_adm_component`"
                         " returning cached output")
                return cached_output
            else:
                log.info("Cache miss for `comparative_regression_adm_component` ..")

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
                                               tags=['regression']))

            # If we get icl_dialog_elements, include them in the
            # dialog, maybe a more explicit argument (wether or not to
            # use icl) makes more sense?
            if icl_dialog_elements:
                if len(icl_dialog_elements[attribute.kdma]) > 0:
                    dialog.extend(icl_dialog_elements[attribute.kdma])

            predict_kdma_prompt = call_with_coerced_args(
                self.prompt_template,
                {'scenario_state': scenario_state,
                 'scenario_description': scenario_description,
                 'choices': choices,
                 'choice_outcomes': {c: None for c in choices},
                 'attribute': attribute.name})

            dialog.append(DialogElement(role='user',
                                        content=predict_kdma_prompt,
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
                    # Choice level reasoning
                    try:
                        attribute_prediction_reasonings[choice].setdefault(
                            attribute.kdma, []).append(response[choice]['reasoning'])
                    # Probe level reasoning
                    except KeyError:
                        attribute_prediction_reasonings[choice].setdefault(
                            attribute.kdma, []).append(response['reasoning'])

            attribute_dialogs[attribute.kdma] = dialog

        outputs = (attribute_prediction_reasonings, attribute_prediction_scores, attribute_dialogs)

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
                       scenario_description_template={_generic_object_repr(self.scenario_description_template)},
                       prompt_template={_generic_object_repr(self.prompt_template)},
                       score_schema_template={_generic_object_repr(self.score_schema_template)},
                       attributes={self.attributes},
                       system_prompt_template={_generic_object_repr(self.system_prompt_template)},
                       num_samples={self.num_samples},
                       enum_scores={self.enum_scores},
                       target_attribute_names_override={self.target_attribute_names_override},
                       reverse_choice_ordering={self.reverse_choice_ordering},
                       )""", flags=re.MULTILINE).strip()


class DeltaRegressionADMComponent(ComparativeRegressionADMComponent):
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
                 enable_caching=False,
                 reverse_choice_ordering=False,
                 output_conflict_resolver=None):
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

        self.reverse_choice_ordering = reverse_choice_ordering

        self.output_conflict_resolver = output_conflict_resolver

    def run(self,
            scenario_state,
            choices,
            icl_dialog_elements=[],
            alignment_target=None):
        if self.reverse_choice_ordering:
            choices = list(reversed(choices))

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

        if self.enable_caching:
            scenario_state_copy = copy.deepcopy(scenario_state)
            if hasattr(scenario_state, 'elapsed_time'):
                # Don't consider the elapsed_time of the state when caching
                scenario_state_copy.elapsed_time = 0

            depends = '\n'.join((
                self.cache_repr(),
                repr(scenario_state_copy),
                repr(choices),
                repr(icl_dialog_elements),
                repr(target_attribute_names)))

            cacher = ub.Cacher('delta_regression_adm_component', depends, verbose=0)
            log.debug(f'cacher.fpath={cacher.fpath}')

            cached_output = cacher.tryload()
            if cached_output is not None:
                log.info("Cache hit for `delta_regression_adm_component`"
                         " returning cached output")
                return cached_output
            else:
                log.info("Cache miss for `delta_regression_adm_component` ..")

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
                                               tags=['regression']))

            # If we get icl_dialog_elements, include them in the
            # dialog, maybe a more explicit argument (wether or not to
            # use icl) makes more sense?
            if icl_dialog_elements:
                if len(icl_dialog_elements[attribute.kdma]) > 0:
                    dialog.extend(icl_dialog_elements[attribute.kdma])

            predict_kdma_prompt = call_with_coerced_args(
                self.prompt_template,
                {'scenario_state': scenario_state,
                 'scenario_description': scenario_description,
                 'choices': choices,
                 'choice_outcomes': {c: None for c in choices},
                 'attribute': attribute.name})

            dialog.append(DialogElement(role='user',
                                        content=predict_kdma_prompt,
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
                    # For now, assign high choice score as difference and low choice score as 0
                    if choice == response['choice']:
                        attribute_prediction_scores[choice].setdefault(
                            attribute.kdma, []).append(response['difference'] / attribute.factor)
                    else:
                        attribute_prediction_scores[choice].setdefault(
                            attribute.kdma, 0)
                    attribute_prediction_reasonings.setdefault(choice, {})
                    # Probe level reasoning
                    attribute_prediction_reasonings[choice].setdefault(
                        attribute.kdma, []).append(response['reasoning'])

            attribute_dialogs[attribute.kdma] = dialog

        outputs = (attribute_prediction_reasonings, attribute_prediction_scores, attribute_dialogs)

        if self.enable_caching:
            cacher.save(outputs)

        return outputs
