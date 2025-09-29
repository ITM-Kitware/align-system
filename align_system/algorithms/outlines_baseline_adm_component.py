import re
import inspect
import copy

from rich.highlighter import JSONHighlighter
import ubelt as ub

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
                 system_prompt=None,
                 num_samples=1,
                 vote_calculator_fn=calculate_votes,
                 enable_caching=False):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template
        self.output_schema_template = output_schema_template

        self.system_prompt_template = system_prompt_template
        self.system_prompt = system_prompt

        self.num_samples = num_samples
        self.vote_calculator_fn = vote_calculator_fn

        self.enable_caching = enable_caching

    def run_returns(self):
        return ('chosen_choice', 'justification', 'dialog')

    def run(self,
            scenario_state,
            choices):
        if self.enable_caching:
            scenario_state_copy = copy.deepcopy(scenario_state)
            if hasattr(scenario_state, 'elapsed_time'):
                # Don't consider the elapsed_time of the state when caching
                scenario_state_copy.elapsed_time = 0

            depends = '\n'.join((
                self.cache_repr(),
                repr(scenario_state_copy),
                repr(choices)))

            cacher = ub.Cacher('outlines_baseline_adm_component', depends, verbose=0)
            log.debug(f'cacher.fpath={cacher.fpath}')

            cached_output = cacher.tryload()
            if cached_output is not None:
                log.info("Cache hit for `outlines_baseline_adm_component`"
                         " returning cached output")
                return cached_output
            else:
                log.info("Cache miss for `outlines_baseline_adm_component` ..")

        scenario_description = call_with_coerced_args(
            self.scenario_description_template,
            {'scenario_state': scenario_state})

        dialog = []
        if self.system_prompt is not None:
            system_prompt = self.system_prompt

            dialog.insert(0, DialogElement(role='system',
                                           content=system_prompt,
                                           tags=['regression']))

        elif self.system_prompt_template is not None:
            system_prompt = call_with_coerced_args(
                self.system_prompt_template, {})

            dialog.insert(0, DialogElement(role='system',
                                           content=system_prompt))

        prompt = call_with_coerced_args(
            self.prompt_template,
            {'scenario_state': scenario_state,
             'scenario_description': scenario_description,
             'choices': choices})

        dialog.append(DialogElement(role='user',
                                    content=prompt))

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

        outputs = (top_choice, top_choice_justification, dialog)

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
            if obj is None:
                return "None"

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
                       output_schema_template={_generic_object_repr(self.output_schema_template)},
                       system_prompt_template={_generic_object_repr(self.system_prompt_template)},
                       system_prompt={self.system_prompt},
                       num_samples={self.num_samples},
                       vote_calculator_fn={_generic_object_repr(self.vote_calculator_fn)},
                       )""", flags=re.MULTILINE).strip()
