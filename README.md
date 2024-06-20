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
[Hydra](https://github.com/NextCenturyCorporation/itm-evaluation-server)
to hand our system configurations.  This allows us to set up sensible
defaults for our configuration, while allowing additional
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
found [here](configs/adm/hybrid_kaleido.yaml)), aligning to
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

#### Experiments

Overriding at the command line is quick and handy, but Hydra has this
notion of "experiments", which are essentially a set of overrides
captured in a new configuration file.  We manage these experiments in
`config/experiments`, and have created an experiment for each of the
delivered ADMs for the Metrics Evaluation (both to run on training
data, and eval data).

## Metrics Evaluation ADM Invocations

Note that to override the API endpoint for the metrics evaluation
ADMs, you can append `interface.api_endpoint='http://127.0.0.1:8080'`
to the command line arguments, setting the value to the correct URL.

### Baseline ADM

```
run_align_system +experiment=metrics_refinement_evaluation/single_kdma_aligned_eval
```

### Aligned ADM 1 (Single KDMA ADM)

```
run_align_system +experiment=metrics_refinement_evaluation/single_kdma_baseline_eval
```

### Aligned ADM 2 (Hybrid Kaleido ADM)

```
run_align_system +experiment=metrics_refinement_evaluation/hybrid_kaleido_eval
```


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
