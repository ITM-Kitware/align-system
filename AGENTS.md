# Agent instructions

This repository holds the `align-system` code, applied research system
code for experimenting with LLM alignment techniques.  When making
code changes, prefer making additive and / or backward compatible
changes.  Regarding coding style, please try to adhere to the
prevailing style of the file(s) you are modifying.  In general prefer
the minimal set of changes that accomplish your goal.  Avoid premature
refactorization.  Prioritize human readability of the code (though
excessive comments are discouraged).

## Core concepts

We use the [Hydra](https://hydra.cc/) configuration library to manage
system and experiment configurations.  The design of the code reflects
this in may places, particularly we make heavy use of object
initialization via Hydra and use "callable" classes in several places
in place of pure functions to support parameterization at config time.

For LLM inference, in many cases we use the
[Outlines](https://dottxt-ai.github.io/outlines/welcome/) library for
enforcing structured output on local model runs, in these cases we
don't need to perform JSON repair or try-catch parsing.

We have an "input_output" JSON file format that's not explicitly
defined anywhere (though you can see the shape of the records in this
file: `align_system/drivers/itm_phase1.py`), this format is important
to downstream tasks and shouldn't be modified arbitrarily.  Any run of
the system should generate an "input_output" JSON file.

We traditionally run algorithms deterministically (to the extent
possible) for the sake of reproducable evaluations.

## Acronyms / Definitions

- Algorithmic Decision Maker (ADM)
- Key-decision Making Attribute (KDMA) -- E.g. "merit focus"
- Alignment Target -- A collection of KDMAs and associated values indicating how an aligned ADM should make decisions
- Baseline ADM -- An ADM that doesn't consider any KDMAs in it's decision making
- Aligned ADM -- An ADM that makes decisions with respect to an alignment "target"
- TA3 -- The evaluation team, responsible for serving up data / scenarios and scoring our ADMs
- TA2 -- Our team, responsible for developing baseline and aligned ADMs
- TA1 -- The KDMA team, responsible for defining the KDMAs and alignment scoring methodologies
- Alignment Score -- A value indicating how close the Aligned ADM's decisions adhere to the alignment target

## Core components

* Interface (e.g. `align_system/interfaces/ta3_caci_action_based_service.py`) -- Provides the scenario / scene data and receives ADM action choices
* ADM (e.g. `align_system/algorithms/pipeline_adm.py`) -- Aligned decision maker, receives state and possible actions and chooses an action.  For an aligned ADM this decision should be with respect to an alignment target.  For a baseline ADM the decision does not consider an alignment target
* Driver (e.g. `align_system/drivers/itm_phase1.py`) -- Connects the interface with the ADM and manages the traversal of the scenario / scene as provided by the interface if needed (i.e. trackes when the scenario changes).  In some cases the driver component also applies some action filtering to prevent the ADM from getting stuck in a loop (i.e. taking the same "no-op" action over and over).  For agentic ADMs (in active development / testing), the driver should supply the ADM with a basic set of tools (i.e. actions the ADM can take in the scenario).  In the future, these tool sets supplied by the driver could be pulled out and considered a new component (perhaps called "Environment")

## Agentic Pipeline Development

An additional focus of this project is expanding the approach of an ADM pipeline to be agentic. 

The agent should be provided a driver/harness (examples of drivers are in align_system/drivers.) They establish a baseline of tools/actions that the agent are able to perform in a given environment. This is done using the Langchain library. 

Harnesses can be expanded with additional tools at configuration time (examples of tools are in /align_system/agent_tools). These are meant to expand the agent's capabilities, and primarily these additional tools are focused on allowing an agent to be aligned towards a task or a KDMA.
