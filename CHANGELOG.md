# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

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
