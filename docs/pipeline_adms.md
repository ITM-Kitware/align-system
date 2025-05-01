# Pipeline ADMs

Decomposing your ADM into a series of steps (or perhaps even a single
step) typically will allow you to only focus on the important
algorithmic code, as we already have several Pipeline ADM steps that
you can re-use to handle much of the data wrangling that may be
necessary.  In this document we'll cover what constitutes a pipeline
ADM, how to create pipeline ADM components (steps), and how to
configure and run them as an ADM.

** Note ** This document assumes you're generally familiar with the
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

In the `RandomChoiceADMComponent`, and with all `ADMComponent`s there
should be at least two methods: `run` which should include the
business logic of the component, and `run_returns` to indicate what to
call the values returned from `run`.

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

This approach attempts to balance component interoperability, and
flexibility.  However this does put some onus on algorithm developers
to pay attention to how outputs and inputs are named, and making sure
these names are consistent across groups of components.

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
example, an ADM Component is only required to have `run`, and
`run_returns` methods.  Beyond that, you'll want to create a default
config for your ADM component,
e.g. `configs/adm_component/direct/random_choice.yaml`.  As a
convention, new ADM components should be put in the
`align_system/algorithms` directory, and should have a filename suffix
of `_adm_component` (e.g. `random_adm_component.py`)

## Configuring 

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

## Existing utility ADM components

These are some ADM components that you may want to re-use for your
pipeline ADM as they handle some of the data wrangling, or are
generally useful.  Some of these are specific to the ITM project,
others are more generic.

- RenameVariablesADMComponent -- Used to rename keys in the working state
- EnsureChosenActionADMComponent -- Ensures that we have a
  `chosen_action` output with a `justification` if available
- ITMFormatChoicesADMComponent -- Populates the list of `choices` for
  ADMs from the properties of the available actions (ITM Specific)
- PopulateChoiceInfo -- Collates some specific working state fields
  for writing out to the "input-output" file format
- ActionParameterCompletionADMComponent -- An ITM specific component
  for ensuring that the chosen action has valid `parameters` based on
  the type of action
- AlignmentADMComponent -- If your ADM is regressing KDMA values, this
  component can be used to select a choice based on those estimated
  KDMA values with respect to an alignment target, and supports
  several alignment functions
