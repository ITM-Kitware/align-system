# ALIGN System

## Setup

### System requirements
It's recommended to run the system on a machine with at least 32GB of
RAM and with a modern GPU with at least 12GB of memory.

### TA3 API
The ALIGN System interfaces with the TA3 ITM MVP web API
(https://github.com/NextCenturyCorporation/itm-mvp), which is
responsible for serving up scenarios and probes, and for handling our
probe responses.  You'll want to have this service installed and
running locally (or on a machine that you can access from wherever
you're running this code).  Instructions for how to do that are
included in that repository's README.

You'll also need to install the client module that's included with
this repository, so ensure that you have this code cloned locally.

### Installation

It's generally recommended to set up a virtual Python environment to neatly manage dependencies (e.g. using `venv` or `conda`).  The `align-system` code can be installed as a Python module with `pip
install git+https://github.com/ITM-Kitware/align-system.git`.

## Running the system

In the Python environment you have set up, a CLI application called `run_align_system` should now be available.  This single entrypoint supports interfacing with both local files on disk, and the TA3 web-based API.  Running the script with `--help` shows which interfaces are available:

```
$ run_align_system --help
usage: run_align_system [-h] {TA3,LocalFiles} ...

ALIGN System CLI

positional arguments:
  {TA3,LocalFiles}  Select interface. Adding --help after interface selection will print interface and system specified arguments
    TA3             Interface with CACI's TA3 web-based service
    LocalFiles      Interface with local scenario / probe JSON data on disk

optional arguments:
  -h, --help        show this help message and exit
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
