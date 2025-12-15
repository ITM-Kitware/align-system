import inspect
import re
import copy

from rich.highlighter import JSONHighlighter
import ubelt as ub

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class PredictMostRelevantADMComponent(ADMComponent):
    def __init__(self,
                 structured_inference_engine,
                 scenario_description_template,
                 prompt_template,
                 relevance_schema_template,
                 attributes={},
                 system_prompt_template=None,
                 num_samples=1,
                 enum_scores=False):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template
        self.relevance_schema_template = relevance_schema_template

        self.attributes = attributes

        self.system_prompt_template = system_prompt_template

        self.num_samples = num_samples
        self.enum_scores = enum_scores

    def run_returns(self):
        return ('attribute_relevance',
                'attribute_dialogs')

    def run(self,
            scenario_state,
            choices,
            # icl_dialog_elements=[],
            alignment_target=None):
        if alignment_target is None:
            target_attribute_names = []
        else:
            target_attribute_names = attributes_in_alignment_target(alignment_target)

        target_attributes = [self.attributes[n] for n in target_attribute_names]
        attribute_names = [attribute.name for attribute in target_attributes]

        attribute_dialogs = {}
        attribute_relevance = {}

        if len(target_attributes) == 1:
            log.info("[bold]*SKIPPING RELEVANCE PREDICTION (only one attribute in target)*[/bold]",
                      extra={"markup": True})
            attribute_relevance[target_attributes[0].kdma]=[1]

        else:
            scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {'scenario_state': scenario_state})

            dialog = []
            if self.system_prompt_template is not None:
                system_prompt = call_with_coerced_args(
                    self.system_prompt_template,
                    {'target_attributes': target_attributes})

                dialog.insert(0, DialogElement(role='system',
                                                content=system_prompt))

            # # If we get icl_dialog_elements, include them in the
            # # dialog, maybe a more explicit argument (wether or not to
            # # use icl) makes more sense?
            # if len(icl_dialog_elements) > 0:
            #     dialog.extend(icl_dialog_elements)

            predict_most_relevant_prompt = call_with_coerced_args(
                self.prompt_template,
                {'scenario_state': scenario_state,
                    'scenario_description': scenario_description,
                    'choices': choices,
                    'attributes': target_attributes})

            dialog.append(DialogElement(role='user',
                                        content=predict_most_relevant_prompt))

            relevance_schema = call_with_coerced_args(
                self.relevance_schema_template,
                {'target_attribute_names': attribute_names})

            dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

            log.info("[bold]*MOST RELEVANT ATTRIBUTE PREDICTION DIALOG PROMPT*[/bold]",
                        extra={"markup": True})
            log.info(dialog_prompt)

            responses = self.structured_inference_engine.run_inference(
                    [dialog_prompt] * self.num_samples, relevance_schema)

            # Use responses to define attribute relevance dictionary
            attribute_relevance = {}
            for target_attribute in target_attributes:
                attribute_relevance[target_attribute.kdma] = []

            for i, response in enumerate(responses):
                log.info("[bold]*RELEVANCE PREDICTION RESPONSE (sample #{})*[/bold]".format(
                    i), extra={"markup": True})
                log.info(response, extra={"highlighter": JSON_HIGHLIGHTER})

                most_relevant = response['most_relevant']
                for target_attribute in target_attributes:
                    if target_attribute.name == most_relevant:
                        attribute_relevance[target_attribute.kdma].append(1)
                    else:
                        attribute_relevance[target_attribute.kdma].append(0)

        log.info("attribute_relevance: {}".format(attribute_relevance), extra={"highlighter": JSON_HIGHLIGHTER})
        return attribute_relevance, attribute_dialogs


class BertRelevanceADMComponent(ADMComponent):
    def __init__(self,
                 scenario_description_template,
                 prompt_template,
                 attributes={},
                 enable_caching=False):
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template
        self.attributes = attributes

        self.enable_caching = enable_caching

    def run_returns(self):
        return ('attribute_relevance')

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

            cacher = ub.Cacher('bert_relevance_adm_component', depends, verbose=0)
            log.debug(f'cacher.fpath={cacher.fpath}')

            cached_output = cacher.tryload()
            if cached_output is not None:
                log.info("Cache hit for `bert_relevance_adm_component`"
                         " returning cached output")
                return cached_output
            else:
                log.info("Cache miss for `bert_relevance_adm_component` ..")

        attribute_relevance = {}
        if len(target_attributes) == 1:
            log.info("[bold]*SKIPPING RELEVANCE PREDICTION (only one attribute in target)*[/bold]",
                      extra={"markup": True})
            attribute_relevance[target_attributes[0].kdma]=[1]

        else:
            scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {'scenario_state': scenario_state})

            if len(icl_dialog_elements) ==0:
                raise RuntimeError('BERT similarity relevance prediction requires ICL examples.')
            else:
                # Get avg similairyt score of this scenario to ICL scenarios
                similarity_scores = {}
                for attribute in target_attributes:
                    scenario_to_match = call_with_coerced_args(
                        self.prompt_template,
                        {'scenario_state': scenario_state,
                        'scenario_description': scenario_description,
                        'choices': choices,
                        'choice_outcomes': {c: None for c in choices},
                        'attribute': attribute.name})

                    example_scenarios = []
                    for icl_dialog_element in icl_dialog_elements[attribute.kdma]:
                        if icl_dialog_element.role == 'user':
                            example_scenarios.append(icl_dialog_element.content)

                    from bert_score import score
                    _, _, F1 = score([scenario_to_match]*len(example_scenarios), example_scenarios, lang="en")
                    similarity_scores[attribute.kdma] = float(F1.sum()/len(F1)) # avg score

                # Pick attribute with highest similarity score as most relevant
                predicted_most_relevant_attribute = max(similarity_scores, key=similarity_scores.get)

                # Set attribute relevance dictionary using prediction
                attribute_relevance = {}
                for target_attribute in target_attributes:
                    if target_attribute.kdma == predicted_most_relevant_attribute:
                        attribute_relevance[target_attribute.kdma] = [1]
                    else:
                        attribute_relevance[target_attribute.kdma] = [0]

        log.info("[bold]*BERT SIMILARITY RELEVANCE PREDICTION*[/bold]",
                 extra={"markup": True})
        log.info("attribute_relevance: {}".format(attribute_relevance),
                 extra={"highlighter": JSON_HIGHLIGHTER})

        if self.enable_caching:
            cacher.save(attribute_relevance)

        return attribute_relevance

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
                       scenario_description_template={_generic_object_repr(self.scenario_description_template)},
                       prompt_template={_generic_object_repr(self.prompt_template)},
                       attributes={self.attributes},
                       )""", flags=re.MULTILINE).strip()
