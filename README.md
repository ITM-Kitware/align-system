# ALIGN System

## Setup

### System requirements

It's recommended to run the system on a machine with at least 32GB of
RAM and with a modern GPU with at least 12GB of memory.

### External Interfaces

The ALIGN System can interface with a few difference services provided
by other teams.  These interfaces may require additional setup
assuming you need to run the services locally for testing / debugging.

#### TA3 Action-based API

The code for the TA3 Action-based service can be found at: [TA3 Evaluation Server API
Repository](https://github.com/NextCenturyCorporation/itm-evaluation-server).

There's a corresponding client module: [TA3 Evaluation Client](https://github.com/NextCenturyCorporation/itm-evaluation-client)

Note that this client module isn't a required dependency for the ALIGN system code.

#### Soartech's TA1 API

Soartech's TA1 service code can be found at: [Soartech's TA1
API](https://github.com/ITM-Soartech/ta1-server-mvp).  This API
provides alignment scores for answered probes and scenarios.

#### ADEPT's TA1 API

ADEPT's TA1 service code can be found at: [ADEPT's TA1
API](https://gitlab.com/itm-ta1-adept-shared/mvp1/adept-api-mvp1).
This API provides alignment scores for answered probes and scenarios.


### Installation

It's generally recommended to set up a virtual Python environment to neatly manage dependencies (e.g. using `venv` or `conda`).  The `align-system` code can be installed as a Python module with `pip
install git+https://github.com/ITM-Kitware/align-system.git`.

## Running the system against the TA3 action-based API

```
$ run_action_based_align_system --help
usage: run_action_based_align_system [-h] {TA3ActionBased} ...

ALIGN Action Based System CLI

positional arguments:
  {TA3ActionBased}  Select interface. Adding --help after interface selection will print interface and system specified arguments
    TA3ActionBased  Interface with CACI's TA3 web-based service

options:
  -h, --help        show this help message and exit
```

Running `--help` after the selected interface prints the full set of options for the interface and system.  E.g.:

```
$ run_action_based_align_system TA3ActionBased --help
usage: run_action_based_align_system TA3ActionBased [-h] [-u USERNAME] [-s SESSION_TYPE] [-e API_ENDPOINT] [--training-session] [-m MODEL] [-t] [-a ALGORITHM] [-A ALGORITHM_KWARGS]
                                                    [--similarity-measure SIMILARITY_MEASURE]

options:
  -h, --help            show this help message and exit
  -u USERNAME, --username USERNAME
                        ADM Username (provided to TA3 API server, default: "ALIGN-ADM")
  -s SESSION_TYPE, --session-type SESSION_TYPE
                        TA3 API Session Type (default: "eval")
  -e API_ENDPOINT, --api_endpoint API_ENDPOINT
                        Restful API endpoint for scenarios / probes (default: "http://127.0.0.1:8080")
  --training-session    Return training related information from API requests
  -m MODEL, --model MODEL
                        LLM Baseline model to use
  -t, --align-to-target
                        Align algorithm to target KDMAs
  -a ALGORITHM, --algorithm ALGORITHM
                        Algorithm to use
  -A ALGORITHM_KWARGS, --algorithm-kwargs ALGORITHM_KWARGS
                        JSON encoded dictionary of kwargs for algorithm initialization
  --similarity-measure SIMILARITY_MEASURE
                        Similarity measure to use (default: 'bert')
```

Here's an example invocation of the system using the TA3 Action-based interface (assuming it's running locally on port `8080`):
```
$ run_action_based_align_system TA3ActionBased \
           -e "http://127.0.0.1:8080" \
           --algorithm "llama_index" \
           --model falcon \
           -s soartech \
           --algorithm-kwargs '{"domain_docs_dir": "/data/shared/MVPData/DomainDocumentsPDF"}'
```

*NOTE* - The first time you run the system it can take upwards of a
half-hour to download the LLM model (which is roughly 25GB).
Subsequent runs of the system should only take a few minutes as the
model is cached.


## Running the system against TA1 services or local files

In the Python environment you have set up, a CLI application called `run_align_system` should now be available.  This single entrypoint supports interfacing with both local files on disk, and the TA3 web-based API.  Running the script with `--help` shows which interfaces are available:

```
$ run_align_system --help
usage: run_align_system [-h] {TA1Soartech,LocalFiles,TA1Adept} ...

ALIGN System CLI

positional arguments:
  {TA1Soartech,LocalFiles,TA1Adept}
                        Select interface. Adding --help after interface selection will print interface and system specified arguments
    TA1Soartech         Interface with Soartech's TA1 web-based service
    LocalFiles          Interface with local scenario / probe JSON data on disk
    TA1Adept            Interface with Adept's TA1 web-based service

options:
  -h, --help            show this help message and exit
```

Running `--help` after the selected interface prints the full set of options for the interface and system.  E.g.:

```
$ run_align_system TA1Soartech --help
usage: run_align_system TA1Soartech [-h] [-s [SCENARIOS ...]] [--alignment-targets [ALIGNMENT_TARGETS ...]] [-e API_ENDPOINT] [-m MODEL] [-t] [-a ALGORITHM] [-A ALGORITHM_KWARGS] [--similarity-measure SIMILARITY_MEASURE]

options:
  -h, --help            show this help message and exit
  -s [SCENARIOS ...], --scenarios [SCENARIOS ...]
                        Scenario IDs (default: 'kickoff-demo-scenario-1')
  --alignment-targets [ALIGNMENT_TARGETS ...]
                        Alignment target IDs (default: 'kdma-alignment-target-1')
  -e API_ENDPOINT, --api_endpoint API_ENDPOINT
                        Restful API endpoint for scenarios / probes (default: "http://127.0.0.1:8084")
  -m MODEL, --model MODEL
                        LLM Baseline model to use
  -t, --align-to-target
                        Align algorithm to target KDMAs
  -a ALGORITHM, --algorithm ALGORITHM
                        Algorithm to use
  -A ALGORITHM_KWARGS, --algorithm-kwargs ALGORITHM_KWARGS
                        JSON encoded dictionary of kwargs for algorithm initialization
  --similarity-measure SIMILARITY_MEASURE
                        Similarity measure to use (default: 'bert')
```


### Example Data

We've included some example scenario, probe, and alignment target data for testing.  These files can be found in the `example_data` directory.  Here's an example system invocation with the provided example files:

```
run_align_system LocalFiles \
    -s example_data/scenario_1/scenario.json \
    --alignment-target-filepath example_data/scenario_1/alignment_target.json \
    -p example_data/scenario_1/probe{1,2,3,4}.json \
    --algorithm "llama_index" \
    --model falcon \
    --algorithm-kwargs '{"domain_docs_dir": "/data/shared/MVPData/DomainDocumentsPDF"}' \
    --align-to-target
```

## ADM Invocations

### Simple Action-based Baseline ADM

Simple baseline (unaligned) system using the `falcon` model:
```
run_action_based_align_system TA3ActionBased \
           --algorithm "llama_index" \
           --model falcon \
           -s soartech \
           --algorithm-kwargs '{"retrieval_enabled": false}' \
           --algorithm "llama_index" \
           --model falcon
```

### Simple Action-based Aligned ADM

Simple aligned system using the `falcon` model (requires domain document PDFs):
```
run_action_based_align_system TA3ActionBased \
           --algorithm "llama_index" \
           --model falcon \
           -s soartech \
           --algorithm-kwargs '{"domain_docs_dir": "/path/to/DomainDocumentsPDF"}' \
           --algorithm-kwargs '{"retrieval_enabled": false}' \
           --algorithm "llama_index" \
           --model falcon \
           --align-to-target
```

### Action-based Chat Baseline ADM

Unaligned system using a Llama 2 chat model:

```
run_action_based_chat_baseline TA3ActionBased \
           -s adept \
           --model meta-llama/Llama-2-13b-chat-hf
```

### Action-based Chat Aligned ADM

Aligned system using a Llama 2 chat model:

```
run_action_based_chat_baseline TA3ActionBased \
           -s adept \
           --model meta-llama/Llama-2-13b-chat-hf \
           --precision half \
           --align-to-target
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

[Creating a custom system script](docs/creating_a_custom_system_script.md)

[Developer environment setup](docs/developer_setup.md)
