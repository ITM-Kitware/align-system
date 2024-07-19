# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Unreleased

### Changed

* Now adding a random UUID suffix to the ADM name parameter when talking to the TA3 server to prevent session clobbering

### Fixed

* Set a limit on the length of output strings in json schemas to avoid running into out of memory errors
* Fixed issue with outlines ADM by catching when target KDMAs are not formatted as dictionaries as expected during eval sessions
* Fixed issue with outlines ADM where responses weren't a list when only a single sample was requested
* Fixed issue with outlines ADM during target KDMA conversion (should only run to_dict on KDMAValue objects)
* Fixed a typo issue with outlines ADM where the positive system prompt was being used instead of the negative system prompt
* Fixed issue with llama3 outlines ADM experiment files where the model wasn't being correctly set

### Added

* Added new implementation of multi-KDMA ADM that regresses KDMA scores based on the outlines structure called `outlines_regression_adm`
* Added regression prompts to `align_system/prompt_engineering/outlines_prompts.py`
* Added KDMA descriptions to `align_system/prompt_engineering/kdma_descriptions.yml`
* Added new [Outlines](https://github.com/outlines-dev/outlines) based structured ADM
* Added outlines based prompts (in `align_system/prompt_engineering/outlines_prompts.py`)
* Added dedicated function to utils for calculating votes (same voting scheme as the single KDMA ADM)
* Added top level config options to force determinism and fix seeds; along with an example experiment to demonstrate
* Added sampler parameter to outlines ADMs (example usage in `align_system/configs/experiment/examples/outlines_sampler.yaml`)
* Added option (on by default) to outlines ADM to filter votes to positive options only, can disable on the command line with `+adm.inference_kwargs.filter_votes_to_positives=False`

### Deprecated 
* The algorithm `align_system/algorithms/chat_kdma_predicting_adm.py` has been replaced by `align_system/algorithms/outlines_regression_adm.py`
* The functionality in `align_system/algorithms/lib/chat/` is no longer being used
* Files `align_system/algorithms/lib/templates/` have been replaced by `align_system/prompt_engineering/`

## 0.4.0

### Changed

* (Major) Changed CLI configuration over to [Hydra](https://hydra.cc/); recommend reading the updated README

### Fixed

* Prevent ADMs from modifying original action objects

### Added

* Added new Oracle ADM (action based; attempts to "choose" best action based on KDMA values)
* Added new action based "Interface" for walking through Input Output JSON files
* Added simple accuracy metrics to the input-output file interface
* Added dedicated docs page for installing external (TA3, TA1s) services

## 0.3.3

### Changed

* Modified the prompt for PulseTaggingADM. Also removed duplicated inference call within `identify_tag_color`
  method. Additionally, removed duplicated RED tag in-context example and replaced with missing BLACK tag
  example.
* Changed default maximization prompt for Kaleido

### Fixed

* Applied attention fixes for Kaliedo provided by UWash
* Fixed an "other choice" ordering issue in Kaleido ADM

### Added

* Added an additional parsing guard in Llama2SinglaKDMAADM
* Added do_sample as an init kwarg for Llama2SinglaKDMAADM (set to False for temperature 0)

## 0.3.2

### Fixed

* Fixed issue where justifications weren't being populated for both Llama2SingleKDMAADM and the HybridKaleidoADM

## 0.3.1

### Added

* Added new Random ADM (action based; chooses random action and action parameters)
* Added additional metrics evaluation candidate ADM configs
* Added logging for final scenario state (alignment scores are provided there in the unstructured field)

### Changed

* Changed the TA3ActionBased interface class to accept a list of scenario IDs to work through (rather than an individual scenario ID)
* No longer restricting the SITREP action based on unvisited and conscious characters

### Fixed

* Fixed issue where Llama2SingleKDMAADM tagging selection could choose an invalid tag
* Not allowing actions that require a character ID to be taken when no characters exist
* Handling rare corner case where generic APPLY_TREATMENT action could be repeated forever
* Fixed mentions of "continuation of care" in maximization prompts

## 0.3.0

### Added

* Added new driver script for TA3 interactions that uses a new YAML config format for ADMs
* Added several ADM config files for new driver script
* Added a new ADM HybridKaleidoADM which defers to a Llama2SingleKDMAADM instance to fill out action parameters
* Added new abstract class for action based ADMs (called ActionBasedADM), requires a `choose_action` method
* Implemented ActionBasedADM `choose_action` method on the KaleidoADM, Llama2SingleKDMAADM, and a new ADM HybridKaleidoADM
* Added alignment accuracy metric in self-evaluation framework
* Added re-usable methods for filling out action parameters to Llama2SingleKDMAADM
* Added short KDMA descriptions for moral deservingness and maximization for Kaleido
* Added new prompt template for selecting the target character of an action
* Added high and low alignment system prompts for SoarTech's maximization KDMA

### Changed

* Replaced instances of "casualties" with "characters" as per the new new TA3 scenario data format
* Changed TA3 interface component over to using TA3 client module (rather than raw HTTP requests)
* Moved the previous `run_align_system.py` script to `run_simplified_align_system.py`, replacing it with the new primary CLI script
* Updated README with respect to new CLI script
* Changed some prompts to not display vitals with a value of None

### Fixed

* Fixed issue with logging of choice scores after multiple-sampling with voting
* Fixed issue where per-sample LLM outputs weren't being logged correctly

## 0.2.6

### Added

* Added bbn pilot data alignability to Single KDMA ADM
* Added compatability for Single KDMA ADM to work with other language models

### Changed

* Moved all system messages into the same directory
* Made number of positive and negative self-consistency votes configurable

### Fixed

* Fixed issue with configurable KDMA Estimator and Distance functions for Kaleido ADM

### Changed

* Better error message on TA3 API action taken failure


## Version 0.2.5

### Added

* Created a multi-comparison-adm

* Created the pulse-tagging-adm

* Added stand-alone llama_index retriever component

* Added retrieval to the llama_2_single_kdma_adm algorithm

### Changed

* Made Llama Index into an ADM that is compatible with the self-evaluation framework by adding a __call__ method


## Version 0.2.4

### Added

* Added Kaleido ADM and dedicated Kaleido CLI script

* Added `partial` option to `format_template` function for partial template completion

* Added `allow_extraneous` option to `format_template` function to ignore extraneous kwargs

### Fixed

* Fixed setting the `loglevel` in CLI scripts


## Version 0.2.3

### Added

* Added --loglevel CLI argument for `run_action_based_chat_baseline.py` script

* Added LanguageModel, ChatLanguageModel classes for ADMs to inherit from

* Added AlignedDecisionMaker interface for ADMs to implement

* Added template system for ADMs to use

* Added evaluation library code to measure ADM performance

* Added ChatKDMAPredictingADM ADM

* Added a few tests for LanguageModel and ChatLanguageModel classes

### Changed

### Fixed

* Fixed issue where TA3 training session flag wasn't being passed to the TA3 API

* Removing training session data info from "action to take" passed to TA3 API


## Version 0.2.2

#### Added

* Added capability to loop over several scenarios in one system run for `run_chat_baseline.py` CLI script

* Added alignment capabilities to `run_chat_baseline.py` CLI script

* Added rich logging capability with the help of the `rich` library

#### Changed


#### Fixed

* Fixed iteration over scenarios / alignment targets with TA1 APIs

* Fixed `--precision` argument in `run_chat_baseline.py` CLI script


## Version 0.2.1

#### Added

* Added aligned decision making capabilities to `llm_chat_baseline.py` algorithm

* Added multiple sampling along with a voting scheme for aligned decision making with the `llm_chat_baseline.py` algorithm

* Added several alignment prompts for MVP2 KDMAs


#### Changed

* Updated action-based chat baseline CLI to use new alignment capabilities

* Changed simple alignment prompt engineering approach to consider a heavy emphasis on a given KDMA when the value is `> 5` (rather than `>= 3`).  This is consistent with how to consider KDMAs with the more sophisticated prompt engineering approach

#### Fixed


## Version 0.2.0

#### Added

* Added llama 2 chat action-based ADM (via new CLI script `run_action_based_chat_baseline`)

* Added llama-index falcon action-based ADM (via new CLI script `run_action_based_align_system`)

* Added support for CACI's new action-based TA3 interface; along with new action-based template CLI script

* Added support for new probe types "PatientOrdering", "SelectTag", and "SelectTreatment"

#### Changed

* Environment now expects Python version >=3.9 (rather than exactly 3.8)

* Deprecated support for old TA3 interface (code not fully removed yet)

* Updated several depedency versions

* Changed BERT implementation to `bert_score` package

#### Fixed


## Version 0.1.1

#### Added

* Added support for Soartech's TA1 web API

* Added support for ADEPT's TA1 web API

* Added Abstract Base Classes for interfaces to help distinguish between the TA3 and TA1 interfaces (which produce alignment scores)

#### Changed

#### Fixed


## Version 0.1.0


#### Added

* Now using poetry to manage dependencies and added `pyproject.toml` and `poetry.lock` in support of this

* Added example template CLI script for custom system interface development along with associated documentation


#### Changed

* Collapsed main CLI scripts into a single script `run_align_system`

* Re-arranged codebase to be pip installable

* Factored out interfaces, for TA3 and local files, into re-usable components

#### Fixed


## Version 0.0.2


#### Added

* Added new heuristic similarity measure and top-level CLI option (`--similarity-measure`) for selecting which similarity measure to use

* Added `--session-type` option to TA3 interface script (`baseline_system.py`)

* Added CPU inference support for llama index algorithm component

* Added support for probes embedded in scenario files for local file interface script (`baseline_system_local_files.py`)

#### Changed

#### Fixed


## Version 0.0.1

* Initial release for MVP demonstration
