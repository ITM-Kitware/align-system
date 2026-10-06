# ALIGN System

## Setup

### System requirements

For local 7B–9B models in half precision, plan for at least 32GB of
RAM and a modern GPU with 24GB of memory. Smaller or quantized models
can use less memory; larger models may require multiple GPUs. See
[System Requirements by Algorithm / Model](#system-requirements-by-algorithm--model)
for model-specific estimates. Hosted API inference does not require a
local GPU or model download.

### Installation

We use [uv](https://docs.astral.sh/uv/) to manage dependencies and the
project's virtual environment. Install uv using its
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/),
then clone the repository and install the locked dependencies with Python 3.12
(the project supports Python 3.10–3.12):

```bash
git clone https://github.com/ITM-Kitware/align-system.git
cd align-system
uv sync --locked --python 3.12
source .venv/bin/activate
```

Run the commands below from the repository root with this environment
activated. Alternatively, prefix them with `uv run`, for example
`uv run run_align_system`. See
[Developer environment setup](docs/developer_setup.md) for optional backend
dependencies.

## Running the system

To run the default sytem configuration against included sample data, simply run:
```
run_align_system
```

*NOTE* - The first run downloads the configured local model. The default
Mistral-7B-Instruct-v0.2 checkpoint is roughly 14.5GB; download time
depends on your connection. Subsequent runs reuse the cached model.


Note that some Hugging Face models are 'gated' and require accepting terms and conditions (e.g. [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)). See [HuggingFace Token Setup](#huggingface-token-setup) for details.

### Hydra

We use
[Hydra](https://hydra.cc/)
to handle our system configurations.  This allows us to set up
sensible defaults for our configuration, while allowing additional
configurations to build up and override existing configs, as well as
override configuration values at runtime.

The default configuration is defined in
[align_system/configs/action_based.yaml](align_system/configs/action_based.yaml):

```yaml
name: action_based

defaults:
  - _self_
  - interface: input_output_file
  - adm: outlines_transformers_structured_baseline
  - driver: itm_phase1
  - override hydra/job_logging: custom

loglevel: "EXPLAIN"

save_log: true
save_raw_log: true
save_input_output: true
save_scoring_output: true
save_alignment_targets: false
save_timing: true
save_last_unstructured_state_per_scenario: false

align_to_target: false
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
run_align_system +experiment=phase1_evaluation/aligned_adm_adept_eval
```

### Aligned ADM SoarTech (Comparative Regression + ICL + Template ADM) (SoarTech eval scenarios)

```
run_align_system +experiment=phase1_evaluation/aligned_adm_soartech_eval
```

## Phase 2 Evaluation (June) ADM Invocations

We've specified Hydra experiments for the Phase 2 Evaluation ADMs.
Note that by default these configurations attempt to connect to
`https://darpaitm.caci.com` as the TA3 API endpoint, but this can be
overridden with `interface.api_endpoint='http://127.0.0.1:8080'` on
the command line.

### Random ADM

(Good candidate for a smoketest)

```
run_align_system +experiment=phase2_june_collab/pipeline_random_live_eval
```

### Baseline ADM

```
run_align_system +experiment=phase2_june_collab/pipeline_baseline_live_eval
```

### Aligned ADM

```
run_align_system +experiment=phase2_june_collab/pipeline_fewshot_comparative_regression_20icl_live_eval
```

### Baseline ADM (multi-KDMA)

While this is the "multi-KDMA" configuration of the baseline, the only
distinction between the single KDMA baseline is the output directory.
Both baseline configurations ignore the targets and KDMAs entirely.

```
run_align_system +experiment=phase2_june_collab/pipeline_baseline_multi_live_eval
```

### Aligned ADM (multi-KDMA):

```
run_align_system +experiment=phase2_june_collab/pipeline_fewshot_comparative_regression_bert_relevance_live_eval
```

## Implementing a new ADM

**Pipeline ADMs:** We have factored out some re-usable ADM components
that can be run step-by-step to act as a complete ADM.  We highly
encourage using the Pipeline ADM framework if you're integrating a new
algorithm, and we intend to migrate older but still relevant ADMs to
Pipeline ADMs.  Please see this dedicated document for Pipeline ADMs
[here](docs/pipeline_adms.md).  The remainder of this section covers
implementing a non-pipeline ADM.

To implement a new ADM, at a minimum you need to implement a class
with a `choose_action` method that takes the following arguments:
- `scenario_state` - Current state of the scenario, model is defined [here](https://github.com/NextCenturyCorporation/itm-evaluation-server/blob/development/swagger_server/models/state.py)
- `available_actions` - List of actions the ADM can choose to take, model is defined [here](https://github.com/NextCenturyCorporation/itm-evaluation-server/blob/development/swagger_server/models/action.py)
- `alignment_target` - Alignment target (or `None` if not aligning), model is defined [here](https://github.com/NextCenturyCorporation/itm-evaluation-server/blob/development/swagger_server/models/alignment_target.py)
- `**kwargs` - A catch all for any additional arguments you want your ADM to receive at inference time

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

The models below are selected by the Hydra configs in
[align_system/configs](align_system/configs), including experiment overrides
and the open-world driver's chat models. Check the resolved Hydra config
for the model and precision used by a particular run.

RAM and GPU figures are planning estimates, not measured minimums. Unless
noted otherwise, local LLM estimates assume FP16/BF16 weights, a small batch,
and a modest context length. The pipeline engines use `precision: half`;
older ADM configs may leave the loading dtype unspecified. Explicit
`precision: full` uses FP32 and doubles weight memory relative to FP16.
Long contexts, parallel requests, and few-shot examples need additional
GPU memory for the KV cache and activations. GPU figures assume the model
fits entirely on GPU; CPU offloading requires additional RAM and is slower.

Disk figures are rounded checkpoint sizes from the linked Hugging Face
repositories (one Transformers weight format), or the linked Ollama tags.
Allow extra space for dependencies, download staging, caches, and outputs;
downloading multiple weight formats or revisions uses more disk. Sizes use
decimal GB. Gated models require the token setup below; consult each model
card for its license, including the base-model terms for fine-tunes.

|Algorithm / configuration|Model / source|RAM (recommended)|GPU memory (planning estimate)|Disk (weights)|Notes|
|---------|-----|---|----------|----------|-----|
|Pipeline ADMs, tagging, demo and evaluation configs|[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)|32GB|18–24GB|~14.5GB|Default for greedy and multinomial pipeline engines; Apache 2.0, ungated.|
|Outlines ADMs, including the default `action_based` ADM|[Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)|32GB in half precision; 64GB in FP32|18–24GB in half precision; 32–48GB in FP32|~14.5GB|Older ADM configs do not all set precision; Apache 2.0, ungated.|
|Decision-flow pipeline, vLLM endpoint, evaluation configs|[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)|32GB|20–24GB|~16.1GB|Default constrained engine; gated, Llama 3.1 license.|
|Phase 1 / multi-KDMA evaluation, integration tests, Phase 2 model comparisons|[Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)|16GB|8–12GB|~6.4GB|Gated, Llama 3.2 license.|
|Phase 2 Spectrum experiments and Spectrum-tuned engine|[spectrum-Llama-3.1-8B-v1](https://huggingface.co/tsor13/spectrum-Llama-3.1-8B-v1)|32GB|20–24GB|~16.1GB|Fine-tuned Llama 3.1 8B; full checkpoint.|
|Phase 2 Spectrum experiments|[spectrum-Qwen3-14B-v1](https://huggingface.co/tsor13/spectrum-Qwen3-14B-v1)|64GB|36–48GB|~29.5GB|Fine-tuned Qwen3 14B; full checkpoint.|
|Phase 2 DeepSeek experiments|[DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)|32GB|20–24GB|~16.1GB|Distilled 8B model; allow extra cache for long reasoning outputs.|
|Phase 2 June / July model comparisons|[DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)|32GB|20–24GB|~15.2GB|Distilled 7B model; allow extra cache for long reasoning outputs.|
|Phase 2 June model comparisons, `persona` ADM config|[gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)|32–64GB|24–32GB|~18.5GB|Gated, Gemma terms.|
|Phase 2 June model comparisons|[Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)|256GB|160GB+ total across GPUs|~141.1GB|Gated; requires model sharding in half precision.|
|Open-world driver, `vllm_qwen25_15b`|[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)|16GB|6–8GB|~3.1GB|The config selects **1.5B**, despite its filename; start vLLM separately.|
|Open-world driver, `vllm_qwen25_3b`|[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)|16GB|8–12GB|~6.2GB|Start vLLM separately; Qwen Research License.|
|Open-world driver, `vllm_qwen25_7b`|[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)|32GB|20–24GB|~15.2GB|Start vLLM separately; Apache 2.0.|
|Open-world driver, `vllm_qwen38_27b`|[Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)|64GB|48GB|~30.9GB|FP8 checkpoint includes BF16 tensors. Config documents text-only serving with `--max-num-seqs 8`; BF16 instead needs an 80GB GPU or two 48GB GPUs.|
|Open-world driver, Ollama|[qwen2.5:7b](https://ollama.com/library/qwen2.5:7b)|16GB|8–12GB|~4.7GB|Q4_K_M tag; [Hugging Face base](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).|
|Open-world driver, Ollama (including ITM prompt variant)|[qwen2.5:32b](https://ollama.com/library/qwen2.5:32b)|32–64GB|24–32GB|~20GB|Q4_K_M tag; [Hugging Face base](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) is ~65.5GB in BF16.|
|Open-world driver, Ollama|[qwen3:32b](https://ollama.com/library/qwen3:32b)|32–64GB|24–32GB|~20GB|Q4_K_M tag; [Hugging Face base](https://huggingface.co/Qwen/Qwen3-32B) is ~65.5GB in BF16.|
|Open-world driver, Ollama|[llama3.1:latest](https://ollama.com/library/llama3.1:latest)|16GB|8–12GB|~4.9GB|Currently the 8B Q4_K_M tag; `latest` may change.|
|Kaleido regression components, hybrid ADM, open-world tool|[tsor13/kaleido-large](https://huggingface.co/tsor13/kaleido-large) / [allenai/kaleido-large](https://huggingface.co/allenai/kaleido-large)|8GB additional|4–8GB additional|~3.1GB|The tsor13 repo redirects to allenai; gated. Loaded in FP32; add these resources to the main LLM's when used together.|
|Older single-KDMA ADMs|[Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)|64GB|32–40GB|~26GB|Still configured by `single_kdma_baseline` / `single_kdma_aligned`; gated, Llama 2 license.|
|Older examples and evaluation configs|[Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) / [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)|32GB|20–24GB|~16.1GB each|Gated, Llama 3 license.|
|Older Phase 1 model comparisons|[Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)|64GB|36–48GB|~27.9GB|14B model, MIT license.|

Hybrid regression also uses [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
(~0.44GB FP32 weights) plus local attribute checkpoints specified in its config.
Budget their memory and disk in addition to the main LLM. Kaleido can also
load [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
for embedding-based deduplication.

The hosted inference configs select `gpt-4o`, `gpt-5.2`,
`claude-haiku-4-5`, `claude-sonnet-4-6`, and `claude-opus-4-6`.
These require network access, provider credentials (`OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`), and API usage charges, but no local model weights
or GPU. A local vLLM or Ollama endpoint still needs the server resources
listed above, even when ALIGN connects through an API.

## HuggingFace Token Setup
Some HuggingFace models are 'gated'. A gated model is indicated by errors like this:
```
Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json.
Access to model meta-llama/Llama-3.1-8B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.
```
1. Visit the model URL while logged in to huggingface to accept the terms and conditions.
2. If not already done, create an access token on the HuggingFace website: click your profile picture > Access Tokens > Create new token > Read > Create Token > copy token
3. Store the token on the system running align:
```
uv run python
>>> from huggingface_hub import login
>>> login()

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

Enter your token (input will not be visible):
Add token as git credential? (Y/n) n
>>>
```

## Quicklinks

[ALIGN App](https://github.com/ITM-Kitware/align-app)

[External interfaces](docs/external_interfaces.md)

[Developer environment setup](docs/developer_setup.md)

[Integration testing](docs/integration_testing.md)

## Citation

If you use the ALIGN system in your research, please cite our paper:

```bibtex
@inproceedings{ravichandran2025align,
  title={ALIGN: Prompt-based Attribute Alignment for Reliable, Responsible, and Personalized LLM-based Decision-Making},
  author={Bharadwaj Ravichandran and David Joy and Paul Elliott and Brian H Hu and Jadie Adams and Christopher Funk and Emily Veenhuis and Anthony Hoogs and Arslan Basharat},
  booktitle={ICML 2025 Workshop on Reliable and Responsible Foundation Models},
  year={2025},
  url={https://openreview.net/forum?id=iQptQH12zD}
}
```

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
