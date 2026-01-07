import re
import inspect
import copy
from functools import lru_cache
from collections.abc import Mapping

import ubelt as ub

from align_system.utils import logging, call_with_coerced_args
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)


# ICL Engines dependent on alignment target, but that could change
# between `run` calls, don't want to reinitialize every time if we
# don't have to
@lru_cache
def init_icl_engine_from_target(icl_generator_partial,
                                attributes,
                                kdma_values):
    log.debug("Initializing ICL generator for target")
    if len(kdma_values) != 1:
        raise RuntimeError("This ADM assumes a single KDMA target, aborting!")

    kdma, value = kdma_values[0]

    target = [{'kdma': kdma,
               'name': attributes[kdma].name,
               'value': value,
               'factor': attributes[kdma].factor,
               'relevant_structured_character_info': attributes[kdma].relevant_structured_character_info}]

    return icl_generator_partial(target_kdmas=target)


class ICLADMComponent(ADMComponent):
    def __init__(self,
                 icl_generator_partial,
                 scenario_description_template,
                 prompt_template,
                 attributes=None,
                 target_attribute_names_override=None,
                 enable_caching=False):
        self.icl_generator_partial = icl_generator_partial
        self.scenario_description_template = scenario_description_template

        if attributes is None:
            attributes = {}
        self.attributes = attributes

        self.prompt_template = prompt_template

        self.target_attribute_names_override = target_attribute_names_override

        self.enable_caching = enable_caching

    def run_returns(self):
        return ('icl_dialog_elements', 'icl_example_info')

    def run(self,
            scenario_state,
            choices,
            actions,
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

        if self.enable_caching:
            scenario_state_copy = copy.deepcopy(scenario_state)
            if hasattr(scenario_state, 'elapsed_time'):
                # Don't consider the elapsed_time of the state when caching
                scenario_state_copy.elapsed_time = 0

            depends = '\n'.join((
                self.cache_repr(),
                repr(scenario_state_copy),
                repr(choices),
                repr(target_attribute_names)))

            cacher = ub.Cacher('icl_adm_component', depends, verbose=0)
            log.debug(f'cacher.fpath={cacher.fpath}')

            cached_output = cacher.tryload()
            if cached_output is not None:
                log.info("Cache hit for `icl_adm_component`"
                         " returning cached output")
                return cached_output
            else:
                log.info("Cache miss for `icl_adm_component` ..")

        # Mapping covers `dict` and `omegaconf.dictconfig.DictConfig`
        if not isinstance(alignment_target, Mapping):
            alignment_target_dict = alignment_target.to_dict()
        else:
            alignment_target_dict = alignment_target

        alignment_target_value_lookup = {
            kdma_values['kdma']: kdma_values['value']
            for kdma_values in alignment_target_dict['kdma_values']
            if 'value' in kdma_values}

        icl_dialog_elements = {}
        icl_example_info = {}

        for attribute in target_attributes:
            icl_dialog_elements[attribute.kdma] = []
            icl_example_info[attribute.kdma] = []

            # Not sure how much this value actually matters for ICL;
            # defaulting to `1.0` if not in the alignment target
            # (e.g. for the "medical" attribute)
            value_for_attribute = alignment_target_value_lookup.get(attribute.kdma, 1.0)

            # Convert alignment target into kdma values (all that's
            # needed for building the icl engines, and need something
            # that's hashable for caching, dicts aren't hashable) as a
            # tuple of tuples; when initializing the icl engine the
            # lru_cache decorator doesn't allow mutable arguments such
            # as lists, need to use tuple
            kdma_values = ((attribute.kdma, value_for_attribute),)

            icl_gen = init_icl_engine_from_target(
                self.icl_generator_partial,
                self.attributes,
                kdma_values)

            scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {'scenario_state': scenario_state,
                 'alignment_target': alignment_target,
                 'attribute': attribute.name,
                 'attributes_of_interest': {attribute.name,}})

            prompt_to_match = call_with_coerced_args(
                self.prompt_template,
                {'scenario_state': scenario_state,
                 'scenario_description': scenario_description,
                 'choices': choices,
                 'choice_outcomes': {c: None for c in choices},
                 'attribute': attribute.name})

            selected_icl_examples = icl_gen.select_icl_examples(
                sys_kdma_name=attribute.kdma,
                scenario_description_to_match=scenario_description,
                prompt_to_match=prompt_to_match,
                state_comparison=scenario_state,
                actions=actions)

            for icl_sample in selected_icl_examples:
                icl_dialog_elements[attribute.kdma].append(DialogElement(role='user',
                                                         content=icl_sample['prompt']))
                icl_dialog_elements[attribute.kdma].append(DialogElement(role='assistant',
                                                         content=str(icl_sample['response'])))

                # Capture ICL example info for choice_info
                icl_info = {
                    'similarity_score': icl_sample['similarity_score'],
                    'prompt': icl_sample['prompt'],
                    'response': icl_sample['response'],
                }
                icl_example_info[attribute.kdma].append(icl_info)

        outputs = (icl_dialog_elements, icl_example_info)

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
                       icl_generator_partial={self.icl_generator_partial},
                       scenario_description_template={_generic_object_repr(self.scenario_description_template)},
                       prompt_template={_generic_object_repr(self.prompt_template)},
                       attributes={self.attributes},
                       target_attribute_names_override={self.target_attribute_names_override},
                       )""", flags=re.MULTILINE).strip()


# ICL Engines dependent on alignment target, but that could change
# between `run` calls, don't want to reinitialize every time if we
# don't have to
@lru_cache
def init_posneg_icl_engines_from_target(icl_generator_partial,
                                        attributes,
                                        kdma_values):
    log.debug("Initializing positive/negative ICL generators for target")
    if len(kdma_values) != 1:
        raise RuntimeError("This ADM assumes a single KDMA target, aborting!")

    kdma, value = kdma_values[0]

    # Assumption here is that KDMA values range from 0-1
    negative_value = 1 - value

    positive_target = [{'kdma': kdma,
                        'name': attributes[kdma].name,
                        'value': value}]

    positive_icl_generator = icl_generator_partial(
        target_kdmas=positive_target)

    negative_target = [{'kdma': kdma,
                        'name': attributes[kdma].name,
                        'value': negative_value}]

    negative_icl_generator = icl_generator_partial(
        target_kdmas=negative_target)

    return positive_icl_generator, negative_icl_generator


class PromptBasedICLADMComponent(ADMComponent):
    def __init__(self,
                 icl_generator_partial,
                 scenario_description_template,
                 prompt_template,
                 attributes=None):
        self.icl_generator_partial = icl_generator_partial
        self.scenario_description_template = scenario_description_template

        if attributes is None:
            attributes = {}
        self.attributes = attributes

        self.prompt_template = prompt_template

    def run_returns(self):
        return 'positive_icl_dialog_elements', 'negative_icl_dialog_elements'

    def run(self,
            scenario_state,
            choices,
            actions,
            alignment_target=None):
        if alignment_target is None:
            raise NotImplementedError(
                "Don't know how to generate positive/negative ICL examples "
                "with no alignment target")
        else:
            target_attribute_names = attributes_in_alignment_target(alignment_target)

        target_attributes = [self.attributes[n] for n in target_attribute_names]

        if len(target_attributes) != 1:
            raise ValueError("Expecting only a single attribute for alignment_target")
        attribute = target_attributes[0]

        scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {'scenario_state': scenario_state,
                 'alignment_target': alignment_target,
                 'attribute': attribute.name,
                 'attributes_of_interest': {attribute.name,}})

        prompt_to_match = call_with_coerced_args(
                self.prompt_template,
                {'scenario_state': scenario_state,
                 'scenario_description': scenario_description,
                 'choices': choices,
                 'choice_outcomes': {c: None for c in choices},
                 'attribute': attribute.name})

        # Mapping covers `dict` and `omegaconf.dictconfig.DictConfig`
        if not isinstance(alignment_target, Mapping):
            alignment_target_dict = alignment_target.to_dict()
        else:
            alignment_target_dict = alignment_target

        # Convert alignment target into kdma values (all that's needed
        # for building the icl engines, and need something that's
        # hashable for caching, dicts aren't hashable)
        kdma_values = tuple((val['kdma'], val['value'])
                            for val in alignment_target_dict['kdma_values'])

        pos_icl_gen, neg_icl_gen = init_posneg_icl_engines_from_target(
            self.icl_generator_partial,
            self.attributes,
            kdma_values)

        pos_selected_icl_examples = pos_icl_gen.select_icl_examples(
            sys_kdma_name=attribute.kdma,
            scenario_description_to_match=scenario_description,
            prompt_to_match=prompt_to_match,
            state_comparison=scenario_state,
            actions=actions)
        pos_icl_dialog_elements = []
        for pos_icl_sample in pos_selected_icl_examples:
            pos_icl_dialog_elements.append(
                DialogElement(role='user',
                              content=pos_icl_sample['prompt']))
            pos_icl_dialog_elements.append(
                DialogElement(role='assistant',
                              content=str(pos_icl_sample['response'])))

        neg_selected_icl_examples = neg_icl_gen.select_icl_examples(
            sys_kdma_name=attribute.kdma,
            scenario_description_to_match=scenario_description,
            prompt_to_match=prompt_to_match,
            state_comparison=scenario_state,
            actions=actions)
        neg_icl_dialog_elements = []
        for neg_icl_sample in neg_selected_icl_examples:
            neg_icl_dialog_elements.append(
                DialogElement(role='user',
                              content=neg_icl_sample['prompt']))
            neg_icl_dialog_elements.append(
                DialogElement(role='assistant',
                              content=str(neg_icl_sample['response'])))

        return pos_icl_dialog_elements, neg_icl_dialog_elements
