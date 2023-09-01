# ALIGN System

## Setup

### System requirements

It's recommended to run the system on a machine with at least 32GB of
RAM and with a modern GPU with at least 12GB of memory.

### External Interfaces

The ALIGN System can interface with a few difference services provided
by other teams.  These interfaces may require additional setup
assuming you need to run the services locally for testing / debugging.

#### TA3 API

The code for the TA3 service can be found at: [TA3 API
Repository](https://github.com/NextCenturyCorporation/itm-mvp).

You'll also need to install the client module that's included with
this repository, so ensure that you have this code cloned locally.

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

## Running the system

In the Python environment you have set up, a CLI application called `run_align_system` should now be available.  This single entrypoint supports interfacing with both local files on disk, and the TA3 web-based API.  Running the script with `--help` shows which interfaces are available:

```
$ run_align_system --help
usage: run_align_system [-h] {TA3,LocalFiles,TA1Soartech,TA1Adept} ...

ALIGN System CLI

positional arguments:
  {TA3,LocalFiles,TA1Soartech,TA1Adept}
                        Select interface. Adding --help after interface selection will print interface and system specified arguments
    TA3                 Interface with CACI's TA3 web-based service
    LocalFiles          Interface with local scenario / probe JSON data on disk
    TA1Soartech         Interface with Soartech's TA1 web-based service
    TA1Adept            Interface with Adept's TA1 web-based service

optional arguments:
  -h, --help            show this help message and exit
```

Running `--help` after the selected interface prints the full set of options for the interface and system.  E.g.:

```
$ run_align_system TA3 --help
usage: run_align_system TA3 [-h] [-u USERNAME] [-s SESSION_TYPE] [-e API_ENDPOINT] [-m MODEL] [-t] [-a ALGORITHM] [-A ALGORITHM_KWARGS] [--similarity-measure SIMILARITY_MEASURE]

optional arguments:
  -h, --help            show this help message and exit
  -u USERNAME, --username USERNAME
                        ADM Username (provided to TA3 API server, default: "ALIGN-ADM")
  -s SESSION_TYPE, --session-type SESSION_TYPE
                        TA3 API Session Type (default: "eval")
  -e API_ENDPOINT, --api_endpoint API_ENDPOINT
                        Restful API endpoint for scenarios / probes (default: "http://127.0.0.1:8080")
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

Here's an example invocation of the system using the TA3 interface:
```
$ run_align_system TA3 -s soartech --algorithm "llama_index" --model falcon --algorithm-kwargs '{"domain_docs_dir": "/data/shared/MVPData/DomainDocumentsPDF"}'
```

*NOTE* - The first time you run the system it can take upwards of a
half-hour to download the LLM model (which is roughly 25GB).
Subsequent runs of the system should only take a few minutes as the
model is cached.

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

### Simple Baseline ADM

Simple baseline (unaligned) system using the `falcon` model:
```
    run_align_system TA3 \
           --algorithm "llama_index" \
           --algorithm-kwargs '{"retrieval_enabled": false}' \
           --model falcon
```

### Simple Aligned ADM

Simple aligned system using the `falcon` model (requires domain document PDFs):
```
    run_align_system TA3 \
           --algorithm "llama_index" \
           --algorithm-kwargs '{"domain_docs_dir": "/path/to/DomainDocumentsPDF"}' \
           --model falcon \
           --align-to-target
```

## System Requirements by Algorithm / Model

*Note: This table is a work-in-progress and will evolve as we add new
algorithms / models*

|Algorithm|Model|RAM|GPU Memory|Disk Space|
|---------|-----|---|----------|----------|
|llama_index|falcon|>32GB|~18GB|~32GB|


## Quicklinks

[Creating a custom system script](docs/creating_a_custom_system_script.md)

[Developer environment setup](docs/developer_setup.md)

[Accessing the Llama-2 models](docs/using_llama_2.md)
