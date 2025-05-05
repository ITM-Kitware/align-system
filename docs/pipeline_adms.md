# Pipeline ADMs

Decomposing your ADM into a series of steps (or perhaps even a single
step) typically will allow you to focus only on the important
algorithmic code, as we already have several Pipeline ADM steps that
you can re-use to handle much of the data wrangling that may be
necessary.  In this document we'll cover what constitutes a pipeline
ADM, how to create pipeline ADM components (steps), and how to
configure and run them as an ADM.

**NOTE** - This document assumes you're generally familiar with the
   system, and how we're using Hydra

## Anatomy of a Pipeline ADM

Let's start with the Hydra config for a random choice ADM:

```YAML
name: pipeline_random

defaults:
  # Import defaults into this namspace (adm) as @name, for further
  # customization

  # ADM components to be used in "steps"
  - /adm_component/misc@step_definitions.format_choices: itm_format_choices
  - /adm_component/direct@step_definitions.random_choice: random_choice
  - /adm_component/misc@step_definitions.random_action_parameter_completion: random_action_parameter_completion
  - /adm_component/misc@step_definitions.ensure_chosen_action: ensure_chosen_action
  - /adm_component/misc@step_definitions.populate_choice_info: populate_choice_info
  # Use definitions in this file to override defaults defined above
  - _self_

instance:
  _target_: align_system.algorithms.pipeline_adm.PipelineADM

  steps:
    # Reference the step instances we want to use in order
    - ${ref:adm.step_definitions.format_choices}
    - ${ref:adm.step_definitions.random_choice}
    - ${ref:adm.step_definitions.random_action_parameter_completion}
    - ${ref:adm.step_definitions.ensure_chosen_action}
    - ${ref:adm.step_definitions.populate_choice_info}
```

Here we've specified several components we'd like to run in our
pipeline.  Generally each of these steps will refer to another config
for each step, these configs are located under the
`configs/adm_component` directory, with further subdirectories for
different categories, e.g. `regression`, `icl`, etc.

For example, here's the contents of the `configs/adm_component/direct/random_choice.yaml` config:

```YAML
_target_: align_system.algorithms.random_adm_component.RandomChoiceADMComponent
```

And the corresponding code (in the file `align_system/algorithms/random_adm_component.py`):

```Python
class RandomChoiceADMComponent(ADMComponent):
    def run_returns(self):
        return 'chosen_choice', 'justification'

    def run(self, choices):
        return random.choice(choices), "Random choice"
```

Since there's nothing to initialize for this class, the configuration
file just refers to the object to initialize (via `_target_`).

In the `RandomChoiceADMComponent`, and with all `ADMComponent`
implementations there should be at least two methods: `run` which
should include the business logic of the component, and `run_returns`
to indicate what to call the values returned from `run`.

The `PipelineADM` class is responsible for running each of the
components under `steps` in serial, and maintaining the working state
of inputs/outputs for the components.  Under the covers, this working
state is just a dictionary with the keys being the names specified by
`run_returns` and the values being the corresponding return values
from `run` calls.  When `run` is called on a pipeline step, the
signature of the method is inspected to determine what variables the
method needs (by name), and those are pulled out of the working state
dictionary and passed along to the `run` call.  For example if we
wanted to use both the `"chosen_choice"` and `"justification"` outputs
from the `RandomChoiceADMComponent` step, our method signature would
look like:

```Python
def run(self, chosen_choice, justification):
    ...
```

This design attempts to balance component interoperability, and
flexibility.  However this does put some onus on algorithm developers
to pay attention to how outputs and inputs are named, and making sure
these names are consistent across groups of components that are
intended to interact.

Since the `PipelineADM` is still an `ActionBasedADM`, and is invoked
as such by the `run_align_system` script, it has a `choose_action`
function with the following signature:

```Python
    def choose_action(self,
                      scenario_state,
                      available_actions,
                      alignment_target=None,
                      **kwargs):
        ...
        return working_output['chosen_action'], working_output
```

`scenario_state`, `available_actions`, and `alignment_target` seed the
working state dictionary.  If `**kwargs` are provided, they're
expanded and added to the working state dictionary as well.

Note that at the end of the `choose_action` function (after all steps
have been run), it's assumed that a `"chosen_action"` record will be
present in the working state (referenced in the code as
`working_output`).  The working state is also returned alongside the
chosen action.

## Adding a Pipeline ADM component

Again referring to the `RandomChoiceADMComponent` as an illustrative
example, an ADM component is only required to have `run`, and
`run_returns` methods.  Beyond that, you'll want to create a default
config for your ADM component,
e.g. `configs/adm_component/direct/random_choice.yaml`.  As a
convention, new ADM components should be put in the
`align_system/algorithms` directory, and should have a filename suffix
of `_adm_component` (e.g. `random_adm_component.py`)

## Configuring your Pipeline ADM

You'll either want to create a new ADM configuration (can follow the
`pipeline_random` config (`configs/adm/pipeline_random.yaml`)), or
override an existing config.  There are a few examples of overriding
existing pipeline ADM configurations in the
`configs/experiment/examples` folder.  Here's the `pipeline_override.yaml` example:

```YAML
# @package _global_
defaults:
  - /adm_component/misc@adm.step_definitions.choice_history: choice_history
  - /adm_component/alignment@adm.step_definitions.cumulative_scalar_alignment: cumulative_avg_dist_scalar
  - override /adm: pipeline_comparative_regression

adm:
  step_definitions:
    choice_history:
      attributes: ${adm.attribute_definitions}

  instance:
    steps:
      # Reference the step instances we want to use in order
      - ${ref:adm.step_definitions.format_choices}
      - ${ref:adm.step_definitions.regression_icl}
      - ${ref:adm.step_definitions.comparative_regression}
      # Retrieve choice_history for alignment
      - ${ref:adm.step_definitions.choice_history}
      - ${ref:adm.step_definitions.cumulative_scalar_alignment}
      # Update choice_history
      - ${ref:adm.step_definitions.choice_history}
      - ${ref:adm.step_definitions.justification_from_reasonings}
      - ${ref:adm.step_definitions.action_parameter_completion}
      - ${ref:adm.step_definitions.ensure_chosen_action}
      - ${ref:adm.step_definitions.populate_choice_info}
```

In this case we're building off of the
`configs/adm/pipeline_comparative_regression.yaml` ADM config.  We add
two new ADM components (added in the `defaults` list):
`choice_history` and `cumulative_avg_dist_scalar`.

Notice that we're setting the `attributes` variable for the
`choice_history` ADM component.  This is also where you can override
variables of steps, even those only defined in the original
`pipeline_comparative_regression` config.

When we add or modify steps (as in this case where we add in
`choice_history`), we do need to specify the full list of steps to be
run (not just the steps that are new or modified).  We suggest using
the ADM config that you're building off of as a reference, to make
sure you don't miss any steps.

## Running your new Pipeline ADM

Pipeline ADMs are given the same treatment as regular ADMs from the
top-level `run_align_system` command.  This means that just setting or
overriding the `adm` field of an existing config or experiment to
point at a pipeline ADM is all it takes.  To run our random pipeline
ADM against a local test file:

```Bash
run_align_system adm=pipeline_random
```

## Comparative Regression Example

Let's look through a real algorithm that's been integrated as a
Pipeline ADM, the comparative regression ADM.  We'll start with the
config file for this ADM, and then work through each of the individual
parts that are new (as compared to the previous example).

```YAML
name: pipeline_comparative_regression

defaults:
  # Import defaults into this namspace (adm) as @name, for further
  # customization

  # Shared variables / components
  - /attribute@mj: moral_judgment
  - /attribute@ib: ingroup_bias
  - /attribute@qol: qol
  - /attribute@vol: vol
  - /inference_engine@structured_inference_engine: outlines_structured_greedy
  - /template/scenario_description@scenario_description_template: with_relevant_char_info
  - /template/prompt@prompt_template: comparative_regression
  - /template/output_schema@comparative_regression_choice_schema: comparative_regression_choice
  # ADM components to be used in "steps"
  - /adm_component/misc@step_definitions.format_choices: itm_format_choices
  - /adm_component/icl@step_definitions.regression_icl: regression
  - /adm_component/regression@step_definitions.comparative_regression: comparative
  - /adm_component/alignment@step_definitions.scalar_alignment: avg_dist_scalar
  - /adm_component/misc@step_definitions.justification_from_reasonings: justification_from_reasonings
  - /adm_component/misc@step_definitions.action_parameter_completion: action_parameter_completion
  - /adm_component/misc@step_definitions.ensure_chosen_action: ensure_chosen_action
  - /adm_component/misc@step_definitions.populate_choice_info: populate_choice_info
  # Use definitions in this file to override defaults defined above
  - _self_

attribute_definitions:
  Moral judgement: ${adm.mj}
  Ingroup Bias: ${adm.ib}
  QualityOfLife: ${adm.qol}
  PerceivedQuantityOfLivesSaved: ${adm.vol}

step_definitions:
  regression_icl:
    scenario_description_template: ${ref:adm.scenario_description_template}
    attributes: ${adm.attribute_definitions}
    prompt_template: ${ref:adm.prompt_template}

  comparative_regression:
    scenario_description_template: ${ref:adm.scenario_description_template}
    prompt_template: ${ref:adm.prompt_template}
    score_schema_template: ${adm.comparative_regression_choice_schema}

instance:
  _target_: align_system.algorithms.pipeline_adm.PipelineADM

  steps:
    # Reference the step instances we want to use in order
    - ${ref:adm.step_definitions.format_choices}
    - ${ref:adm.step_definitions.regression_icl}
    - ${ref:adm.step_definitions.comparative_regression}
    - ${ref:adm.step_definitions.scalar_alignment}
    - ${ref:adm.step_definitions.justification_from_reasonings}
    - ${ref:adm.step_definitions.action_parameter_completion}
    - ${ref:adm.step_definitions.ensure_chosen_action}
    - ${ref:adm.step_definitions.populate_choice_info}
```

While this ADM has some similarities with the random ADM example, in
that there are steps (even a few that are shared) defined and
enumerated.  There are few more pieces here, which we'll cover in
detail in subsequent subsections.

### Attribute Definitions

Since this is an
alignable ADM, it should have some awareness (and metadata) about the
attributes it's capable of aligning to.  To that end, we include some
attribute definitions for each of the KDMAs this ADM is prepared to
work with.  The attribute definitions are pulled from atrribute
configs, as an example, here's the contents of the "Moral judgement"
attribute config:

```YAML
_target_: align_system.data_models.attribute.Attribute

name: moral judgment
kdma: "Moral judgement"
description: Moral judgment is the tendency to consider moral deservingness or culpability, and to reward moral deservingness. Someone with high moral judgment will consider a patient's intent to harm others and degree of responsibility for a situation when making medical triage decisions. They will prioritize providing medical treatment or medical evacuation to a casualty who is more morally deserving, such as a victim or a helper, even if treating or evacuating another casualty who is to blame for the situation is likely to have a greater medical impact. Someone with low moral judgment will not consider a patient's moral deservingness when making medical triage decisions.

factor: 100
score_examples: 'Treating someone who acted with no intent and had an indirect cause of harm would score a {{ "%g" | format(1.0 * kdma_scale_factor) }}. Treating someone who acted to intend minor help and had an indirect cause of harm would score a {{ "%g" | format(0.8 * kdma_scale_factor) }}. Treating someone who acted with no intent and had a somewhat indirect cause of harm would score a {{ "%g" | format(0.4 * kdma_scale_factor) }}. Treating someone who acted to intend minor harm and had a somewhat indirect cause of harm would score a {{ "%g" | format(0.3 * kdma_scale_factor) }}. Treating someone who acted to intend major harm and had a direct cause of harm would score a {{ "%g" | format(0 * kdma_scale_factor) }}.'
valid_scores:
  _target_: align_system.data_models.attribute.AttributeValidValueRange

  min: 0
  max: 100
  step: 1
  
relevant_structured_character_info: ['intent', 'directness_of_causality', 'injuries']
```

We also set up an `attribute_definitions` object in the config (it's
path will be `adm.attribute_definitions`) which contains all of the
attributes we've included.  Several components in this ADM will make
use of the `attribute_definitions`.

### Inference Engines

Another piece we've defined in the comparative regression ADM config
is an inference engine, or in this case a structured inference engine.
This piece is responsible for loading / managing the LLM instance, and
running inference.  Since this is a structured inference engine, we
provide it with an output schema and the generated output should be
structured (and adhere to the schema).  The config for the inference
engine being used in this case:

```YAML
_target_: align_system.algorithms.outlines_inference_engine.OutlinesTransformersInferenceEngine

model_name: mistralai/Mistral-7B-Instruct-v0.3
precision: half
sampler:
  _target_: outlines.samplers.GreedySampler
```

Note that the class here is `OutlinesTransformersInferenceEngine`, and
we're using the `"mistralai/Mistral-7B-Instruct-v0.3"` model at half
precision.  And using the `outlines` greedy sampler.

The structured inference engine will be used by more than one
component in the pipeline, but we don't want multiple instances of it
as the LLMs take up quite a bit memory.  To only initialize and refer
to a single instance, the structured inference engine will be referred
to with the following variable:
`${ref:adm.structured_inference_engine}`.  The `ref:` prefix is a
custom variable interpolation type that has special handling in the
`align-system` codebase to only point to a single instantiated
instance.

Note that in the top-level configuration file for this ADM, there
aren't any such reference to the structured inference engine.  These
are actually set in the adm component configs themselves (but they
still point to what's defined at the ADM level).  For example, in the
`adm_component/regression/comparative.yaml` config we have:

```YAML
_target_: align_system.algorithms.comparative_regression_adm_component.ComparativeRegressionADMComponent

structured_inference_engine: ${ref:adm.structured_inference_engine}
num_samples: 1
attributes: ${ref:adm.attribute_definitions}
system_prompt_template:
  _target_: align_system.prompt_engineering.outlines_prompts.ComparativeKDMASystemPrompt
```

### Shared Templates

Along with the inference engine being shared between components, it's
not uncommon to want to share text templates or prompts between
different components.  In the comparative regression example, we want
to share the same scenario description and prompt templates between
the in-context learning component, and the regression component.  This
is accomplished in the same way as the structured inference engine and
the attributes; they're defined for the ADM, and then referenced by
the individual components that need them.  You can see this in action
in the `step_definitions` section of the comparative regression
pipeline config:

```YAML
...
step_definitions:
  regression_icl:
    scenario_description_template: ${ref:adm.scenario_description_template}
    attributes: ${adm.attribute_definitions}
    prompt_template: ${ref:adm.prompt_template}

  comparative_regression:
    scenario_description_template: ${ref:adm.scenario_description_template}
    prompt_template: ${ref:adm.prompt_template}
    score_schema_template: ${adm.comparative_regression_choice_schema}

...
```

The `step_definitions` is typically how you'll want to override
variables / parameters for the `adm_components` you're using.

### Steps

As with the random ADM example, now that we've defined everything, we
just enumerate the steps for our PipelineADM instance.

```YAML
...
instance:
  _target_: align_system.algorithms.pipeline_adm.PipelineADM

  steps:
    # Reference the step instances we want to use in order
    - ${ref:adm.step_definitions.format_choices}
    - ${ref:adm.step_definitions.regression_icl}
    - ${ref:adm.step_definitions.comparative_regression}
    - ${ref:adm.step_definitions.scalar_alignment}
    - ${ref:adm.step_definitions.justification_from_reasonings}
    - ${ref:adm.step_definitions.action_parameter_completion}
    - ${ref:adm.step_definitions.ensure_chosen_action}
    - ${ref:adm.step_definitions.populate_choice_info}
```

## Existing utility ADM components

These are some ADM components that you may want to re-use for your
pipeline ADM as they handle some of the data wrangling, or are
generally useful.  Some of these are specific to the ITM project,
others are more generic.

- **RenameVariablesADMComponent** - Used to rename keys in the working state
- **EnsureChosenActionADMComponent** - Ensures that we have a
  `chosen_action` output with a `justification` if available
- **ITMFormatChoicesADMComponent** - Populates the list of `choices` for
  ADMs from the properties of the available actions (ITM Specific)
- **PopulateChoiceInfo** - Collates some specific working state fields
  for writing out to the "input-output" file format
- **ActionParameterCompletionADMComponent** - An ITM specific component
  for ensuring that the chosen action has valid `parameters` based on
  the type of action
- **AlignmentADMComponent** - If your ADM is regressing KDMA values, this
  component can be used to select a choice based on those estimated
  KDMA values with respect to an alignment target, and supports
  several alignment functions
