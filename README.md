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

### Python environment and dependencies

We recommend setting up a virtual Python environment to neatly manage
dependencies.  For example, using conda:

```
conda create -n align-system python=3.10
```

Activate the environment:
```
conda activate align-system
```

Then, install the required dependencies:
```
pip install -r requirements.txt
```

Then, install the TA3 ITM MVP client module:
```
pip install -e /path/to/itm-mvp/itm_client
```

## Running the system

In the Python environment you have set up, the `baseline_system.py`
script is the entrypoint for running the system.  Help text is shown
below:

```
$ python baseline_system.py
usage: baseline_system.py [-h] [-e API_ENDPOINT] [-u USERNAME] [-m MODEL] [-t] [-a ALGORITHM] [-A ALGORITHM_KWARGS]

Simple LLM baseline system

options:
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
```

An example invocation of the system:
```
$ python baseline_system.py --model gpt-j
```

*NOTE* - The first time you run the system it can take upwards of a
half-hour to download the LLM model (which is roughly 25GB).
Subsequent runs of the system should only take a few minutes as the
model is cached.

## ADM Invocations

### Simple Baseline ADM

Simple baseline (unaligned) system using the `falcon` model:
```
    python baseline_system.py \
           --algorithm "llama_index" \
           --algorithm-kwargs '{"retrieval_enabled": false}' \
           --model falcon
```

### Simple Aligned ADM

Simple aligned system using the `falcon` model (requires domain document PDFs):
```
    python baseline_system.py \
           --algorithm "llama_index" \
           --algorithm-kwargs '{"domain_docs_dir": "/path/to/DomainDocumentsPDF"}' \
           --model falcon \
           --align-to-target
```
