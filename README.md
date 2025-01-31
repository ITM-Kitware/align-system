# ALIGN System

## Setup

### System requirements

It's recommended to run the system on a machine with at least 32GB of
RAM and with a modern GPU with at least 12GB of memory.

### Installation

It's generally recommended to set up a virtual Python environment to neatly manage dependencies (e.g. using `venv` or `conda`).  The `align-system` code can be installed as a Python module with `pip
install git+https://github.com/ITM-Kitware/align-system.git`.

## Running the system

To run the default sytem configuration against included sample data, simply run:
```
run_align_system
```

*NOTE* - The first time you run the system it can take upwards of a
half-hour to download the LLM model (which is roughly 25GB).
Subsequent runs of the system should only take a few minutes as the
model is cached.

### Hydra

We use
[Hydra](https://hydra.cc/)
to handle our system configurations.  This allows us to set up
sensible defaults for our configuration, while allowing additional
configurations to build up and override existing configs, as well as
override configuration values at runtime.

The default configuration is (note that Hydra configuration files are `.yaml`):
```
name: action_based

defaults:
  - interface: input_output_file
  - adm: single_kdma_baseline
  - override hydra/job_logging: custom

loglevel: "EXPLAIN"

save_log: true
save_input_output: true
save_scoring_output: true

align_to_target: False
```

#### Overriding at runtime

Hydra's override syntax on the command line is fairly straightforward
(covered in their documentation
[here](https://hydra.cc/docs/advanced/override_grammar/basic/)).
Though note the `+` prefix for `+alignment_target=maximization_high`
in the example below, here we're adding a new configuration field that
isn't specified in the default configuration (as opposed to overriding
an existing field)

In the example below, we're building upon the default configuration,
but we're running the `kaleido_hybrid` ADM (it's configuration can be
found [here](align_system/configs/adm/hybrid_kaleido.yaml)), aligning to
`maximization_high`, and interfacing with the `ta3` service (instead
of a local sample file).

```
run_align_system \
    loglevel="DEBUG" \
    adm=hybrid_kaleido \
    +alignment_target=maximization_high \
    align_to_target=true \
    interface=ta3 \
    interface.session_type='soartech' \
    interface.scenario_ids='["desert-1-train1","jungle-1-train1","submarine-1-train1","urban-1-train1"]' \
    interface.training_session=true
```

#### Outputs

By default, the `run_align_system` command puts output files in the
current working directory, under
`outputs/<year-month-day>/<hour-minute-second>`
(e.g. `"outputs/2024-06-18/14-55-31"`).  The output directory and
sub-directory pattern can be overridden on the command line by
settting the `hydra.run.dir` parameter.

```
run_align_system hydra.run.dir='my_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
```

Hydra also saves out all of the config parameters, config overrides,
and internal hydra parameters for the run in the output directory in a
subdirectory called `.hydra`.

#### Output scores

Assuming the `save_scoring_output` configuration option is `true`
(this is the default), and you're not running against the TA3 server
for an `eval` session, the `run_align_sytem` command will save any
scoring output from the run as `scores.json`.

#### Experiments

Overriding at the command line is quick and handy, but Hydra has this
notion of "experiments", which are essentially a set of overrides
captured in a new configuration file.  We manage these experiments in
`align_system/configs/experiment`, and have created an experiment for each of the
delivered ADMs for the Metrics Evaluation (both to run on training
data, and eval data).

## Phase 1 Evaluation ADM Invocations

We've specified Hydra experiments for the Phase 1 Evaluation ADMs.
Note that by default these configurations attempt to connect to
`https://darpaitm.caci.com` as the TA3 API endpoint, but this can be
overridden with `interface.api_endpoint='http://127.0.0.1:8080'` on
the command line.

### Random ADM

(Good candidate for a smoketest)

```
run_align_system +experiment=phase1_evaluation/random_eval_live
```

### Baseline ADM

```
run_align_system +experiment=phase1_evaluation/baseline_eval_live
```

### Aligned ADM Adept (Comparative Regression + ICL + Template ADM) (ADEPT eval scenarios)

```
run_align_system +experiment=dry_run_evaluation/aligned_adm_adept_eval
```

### Aligned ADM SoarTech (Comparative Regression + ICL + Template ADM) (SoarTech eval scenarios)

```
run_align_system +experiment=dry_run_evaluation/aligned_adm_soartech_eval
```


## Implementing a new ADM

To implement a new ADM, at a minimum you need to implement a class
with a `choose_action` method that takes the following arguments:
- `scenario_state` -- Current state of the scenario, model is defined [here](https://github.com/NextCenturyCorporation/itm-evaluation-server/blob/development/swagger_server/models/state.py)
- `available_actions` -- List of actions the ADM can choose to take, model is defined [here](https://github.com/NextCenturyCorporation/itm-evaluation-server/blob/development/swagger_server/models/action.py)
- `alignment_target` -- Alignment target (or `None` if not aligning), model is defined [here](https://github.com/NextCenturyCorporation/itm-evaluation-server/blob/development/swagger_server/models/alignment_target.py)
- `**kwargs` -- A catch all for any additional arguments you want your ADM to receive at inference time

And this `choose_action` method should return one of the
`available_actions`, which may require filling in additional
parameters such as the treatment location for a treatment action, or
triage tag category for a tagging action

The [RandomADM](align_system/algorithms/random_adm.py) is a good
example to start with.

### Creating a configuration file for your new ADM

To run your new ADM from the command line, you'll need to create a
default configuration file in the `align_system/configs/adm`
directory.  The name of the config file you create is important, as
that's how you'll reference your ADM from the command line.

As an example, here's the `single_kdma_aligned.yaml` config:

```
instance:
  _target_: align_system.algorithms.llama_2_single_kdma_adm.Llama2SingleKDMAADM

  hf_model: meta-llama/Llama-2-13b-chat-hf
  precision: half
  temperature: 0.7

inference_kwargs:
  baseline: false
  n_negative_samples: 5
  n_positive_samples: 5
  shuffle: true
```

Notice that there are two top level keywords, `instance` (for
specifying how an instance of your ADM should be created), and
`inference_kwargs` (will be passed to your ADM's `choose_action`
method as the `**kwargs` at inference time)

The `_target_` field under `instance` should be the full import path
to your ADM's class.  Note that your ADM doesn't have to be a class in
the `align_system` module, as long as it's importable.

To use your new ADM on the command line, do `run_align_system
adm=my_new_adm` (assuming you named your new ADM config file
`my_new_adm.yaml`).

## System Requirements by Algorithm / Model

*Note: This table is a work-in-progress and will evolve as we add new
algorithms / models*

|Algorithm|Model|RAM|GPU Memory|Disk Space|Hugging Face Link|Notes|
|---------|-----|---|----------|----------|-----------------|-----|
|llama_index|tiiuae/falcon-7b-instruct|>32GB|~18GB|~13GB|https://huggingface.co/tiiuae/falcon-7b-instruct||
|llm_chat|Llama-2-7b-chat-hf|>32GB|~18GB|~13GB|https://huggingface.co/meta-llama/Llama-2-7b-chat-hf|Requires license agreement: https://ai.meta.com/llama/license/|
|llm_chat|Llama-2-13b-chat-hf|>48GB|~28GB|~25GB|https://huggingface.co/meta-llama/Llama-2-13b-chat-hf|Requires license agreement: https://ai.meta.com/llama/license/|


## Quicklinks

[External interfaces](docs/external_interfaces.md)

[Developer environment setup](docs/developer_setup.md)

[Integration testing](docs/integration_testing.md)

## Acknowledgments

This research was developed with funding from the Defense Advanced
Research Projects Agency (DARPA) under Contract
No. FA8650-23-C-7316. The views, opinions and/or findings expressed
are those of the author and should not be interpreted as representing
the official views or policies of the Department of Defense or the
U.S. Government.

## Disclaimer

We emphasize that our work should be considered academic research, as
we cannot fully guarantee model outputs are free of inaccuracies or
biases that may pose risks if relied upon for medical
decision-making. Please consult a qualified healthcare professional
for personal medical needs.
