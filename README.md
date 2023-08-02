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

The `align-system` code can be installed as a Python module with `pip
install git+https://github.com/ITM-Kitware/align-system.git`.

It's generally recommended to set up a virtual Python environment to
neatly manage dependencies.  For example, using `venv`:

```
python3.8 -m venv venv
```

This creates a new directory called `venv`.  You then activate this
new environment with:

```
source venv/bin/activate
```

#### Environment setup with Conda

It may be easier to create an environment using Conda (or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)) if you
require a different version of Python than what's on the system and
don't have `sudo` permissions.  In that case you can create an
environment with a specific Python version with:

```
conda create -n align-system python=3.8
```

Then activate the environment with:
```
conda activate align-system
```

### Developer Installation

If you're working directly on the `align-system` code, we recommend
using [Poetry](https://python-poetry.org/) as that's what we use to
manage dependencies.  Once poetry is installed, you can install the
project (from inside a local clone of this repo) with `poetry
install`.  By default poetry will create a virtual environment (with
`venv`) for the project if one doesn't already exist.


## Running the system

In the Python environment you have set up, two CLI applications are available: `align_baseline_system` (interfaces with the TA3 server) and `align_baseline_system_local_files` (works with local files on disk).  They have similar command line arguments (can be shown with the `--help` argument), but we'll just demonstrate how to run the `align_baseline_system` script here:

```
$ align_baseline_system --help
usage: align_baseline_system [-h] [-e API_ENDPOINT] [-u USERNAME] [-m MODEL] [-t] [-a ALGORITHM] [-A ALGORITHM_KWARGS] [--similarity-measure SIMILARITY_MEASURE] [-s SESSION_TYPE]

Simple LLM baseline system

optional arguments:
  -h, --help            show this help message and exit
  -e API_ENDPOINT, --api_endpoint API_ENDPOINT
                        Restful API endpoint for scenarios / probes (default: "http://127.0.0.1:8080")
  -u USERNAME, --username USERNAME
                        ADM Username (provided to TA3 API server, default: "ALIGN-ADM")
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
  -s SESSION_TYPE, --session-type SESSION_TYPE
                        TA3 API Session Type (default: "eval")
```

An example invocation of the system:
```
$ align_baseline_system --model gpt-j
```

*NOTE* - The first time you run the system it can take upwards of a
half-hour to download the LLM model (which is roughly 25GB).
Subsequent runs of the system should only take a few minutes as the
model is cached.

## ADM Invocations

### Simple Baseline ADM

Simple baseline (unaligned) system using the `falcon` model:
```
    align_baseline_system \
           --algorithm "llama_index" \
           --algorithm-kwargs '{"retrieval_enabled": false}' \
           --model falcon
```

### Simple Aligned ADM

Simple aligned system using the `falcon` model (requires domain document PDFs):
```
    align_baseline_system \
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
