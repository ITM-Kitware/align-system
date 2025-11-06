# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Unreleased

### Added

* Added more details and examples to the pipeline ADM documentation
* Added a new Phase 2 medical urgency alignment function (weighted) to reflect program collaboration updates
* Added option for ICL example choice ordering - fixed, swapped, or random
* Added option to swap choice ordering for comparative regression pipeline ADM component
* Added capability to resolve pipeline ADM step output conflicts with a custom function
* Added inference engine for spectrum tuned LLMs that appropriately reformats chat template roles

## 0.5.9

### Added

* Added new (optional) domain argument for the TA3 interface
* Added support for the TA3 interface domain `p2triage`
* Added `state_hydration_domain` to `IncontextExampleGenerator` and `InputOutputFileInterface`. Providing a value of `None` or `p1` results in Phase 1 behavior while a value of `p2triage` is meant for hydrating states from the corresponding domain on the TA3 server.
* Added multi-KDMA evaluation support with dedicated baseline multi config and BERT-relevance config for Phase 2 June evaluation
* Added configurations for Phase 2 June evauation
* Added caching functionality for baseline and comparative regression ADM components
* Added ICL-based relevance prediction ADM with BERT similarity
* Added least_similar_examples option for ICL
* Added Phase 2 midpoint based alignment function (with relevance support) along with unit tests
* Added medical-only alignment function
* Added dedicated Phase 2 regression component config for Kaleido
* Added Kaleido ADM experiment config variants including a "mashup" for Phase 2
* Added pipeline components: relevance aggregation, regression oracle, and relevance oracle
* Added `state_hydration_domain` option to ICL and input/output interface
* Added pipeline component for post-hoc rule-based regression value correction
* Added new KDMAs for Phase 2 June collaboration: "personal_safety" and "search"
* Added local copies of Phase 1 data models from TA3 code to support backward compatability
* Added Phase 2 alignment target config files
* Added `ubelt` dependency for caching support

### Changed

* Changed some Phase 1 components and templates to point at local Phase 1 data models

## 0.5.8

### Fixed

* Fixed non-determinism issues with outlines_adm baseline with shuffle_choices arg and outlines seed
* Fixed configs referenced in top-level README for Phase 1 Evaluation runs

### Added

* Added PipelineADM and many pipeline ADM components, configs, and integration tests
* Added new data models for Attributes and Dialogs
* Added `call_with_coerced_args` utility function to handle calling functions with different input requirements
* Added documentation for new pipeline ADM developement: [Pipeline ADMs](docs/pipeline_adms.md)
* Added a specialized Hydra instantiation function to support shared objects
* Added shuffle_choices inference_kwarg argument for outlines_adm baseline
* Added option to set outlines RNG seed for outlines_adm

## 0.5.7

### Fixed

* Fixed alignment target parsing for KaleidoADM
* Fixed issue where action parameters aren't set if there is only 1 remaining heuristic option.
* Fixed crash when using `save_last_unstructured` flag without alignment target

### Added

* Added raw log output file (on by default) in addition to rich formatted log output file
* Added choice_info output to KaleidoADM to support MSE analysis
* Added multi-KDMA alignment targets for testing
* Added Outlines based Personas ADM
* Added dedicated utility function for inferring alignment target type
* Added multi-KDMA alignment function that weights distance by relevance
* Added a relevance oracle ADM
* Added an optional relevance prediction step with ICL to the comparative regression ADM
* Added options to use cumulative KDE and/or relevance weighted alignment function to Kaleido ADM
* Added integration testing script (`tests/run_integration_test.py`) and associated configs and data files
* Added basic support for NAACL24 and OpinionQA datasets
* Initial API support for ALIGN-App

### Changed

* Changed KaleidoADM prompt log output to "info" instead of "debug"
* Exposed `get_dialogs` as a static method for `outlines_adm.py` to support ALIGN-App


## 0.5.6

### Changed

* Updated Phase 1 experiment configs for final Phase 1 Eval delivery

## 0.5.5

### Added

* Added Phase 1 Evaluation experiment configuration files
* Added ICL example selection method that gives larger weight to examples with the same characetr ids as the current probe. To use set `incontext.method` to `matching_characters`.
* Added ICL example selection method that gives larger weight to examples with the same action types as the current probe. To use set `incontext.method` to `matching_actions`.
* Added retrieved ICL examples to input-output.json

## 0.5.4

### Changed

* Changed `incontext` `normalization` setting to be off (null/rawscores)
* ``incontext.leave_one_out=false`` should now be configured as ``incontext.leave_one_out_strategy=null``. Default behavior is **no** leave one out behavior.
  Previous ``incontext.leave_one_out=true`` should be specified as ``incontext.leave_one_out_strategy=scenario_description``. Additionally, duplicate ICL examples,
  based on the chosen similiarity strategy, are now removed.
* Changed `training_session` flag for TA3 interface from boolean to string (expecting "full" or "solo" or None)
* Changed the comparative regression prompt to only include the structured chararcter information listed in `relevant_structured_character_info` in `kdma_descriptions.yaml`. To include all strucutured information that is unique across characters in the prompt (as was previously done automatically), specify `relevant_structured_character_info = ['all_unique']`.
* Improved the QoL `description` and `score_examples` in `kdma_descriptions.yaml`
* Changed default treatment parameter selection to use heuristic treatment options
* Updated to transformers>=4.46.2 (and added necessary dependencies) to support newer models

### Added

* Added an option for sorting incontext examples responses: `incontext.sort_actions`
* Added character-based leave one out option: ``incontext.leave_one_out_strategy=characters``
* Phase 1 experiments directory
* Added the option to filter out TAG CHARACTER responses by setting `filter_tag_character` to true
* Added a history-based alignment function for scalar targets that uses distance to a running mean. To use specify `inference_kwargs.distribution_matching` as `cumulative_average`
* Added the option to enumerate the valid regression scores in the json schema by specifying `inference_kwargs.enum_scores` as true. Valid score options for each KDMA are added to `align_system/prompt_engineering/kdma_descriptions.yml`.
  Valid score options may be specifed as a list via ``values``, or a ``range`` specifed as dictionary of ``min`` (inclusive), ``max`` (inclusive), ``step``
* Added option to configure ICL example ordering: ``incontext.most_similar_first=true`` for the most similar ICL example first, ``false`` for most similar ICL example last.
* Added the option to normalize KDE targets based on prior data. To use, set `adm.inference_kwargs.kde_norm=priornorm` and `adm.inference_kwargs.priornorm_factor` to the normalization weight you want (1 is fully normalized, 0 is no normalization or `rawscores`, default is 0.5.
* Added KDMA scaling factor option. Scale factors for each KDMA are added to `align_system/prompt_engineering/kdma_descriptions.yml`
* Added heuristic treatment options component
* Added incontext examples to the `input_output.json` files for comparative regression

### Fixed

* Fixed issue where choice history was persisting across scenarios -- supporting new optional method for ADMs `reset_history` called at the start of each new scenario

## 0.5.3

### Changed

* Moved incontext learning functionality into `incontext_utils.py` and updated the base outlines and comparative regession ADMS to use this module.
* Moved the `format_choices()` function from the `OutlinesTransformersADM` class in `outlines_adm.py` to a new utils file: `adm_utils.py` so it can be used across ADMs.
* Update example_data/input_output_files to use DRE training scenarios
* Changed default config to use `outlines_transformers_structured_baseline` (rather than the older `single_kdma_baseline`)
* Adjusted `choose_action()` to enable returning an ADM-specific `choice_info` dictionary that is written to the resulting `input_output.json` file
* When alignment target is optionally saved out in `run_align_system` save as JSON instead of YAML

### Added

* Added option to normalize KDMA values in incontext examples
* Added a probabilistic option to alignment utilities. Exposed this option in oracle, comparative regression, and
  hybrid regression ADMs.
* Example config for deterministic outlines-based ADM runs (`align_system/configs/experiment/examples/outlines_force_determinism.yaml`). Requires setting `force_determinsim` to true and using greedy sampler.
* Added a history-based/cumulative KDE option to alignment utilities. Exposed this option in oracle and comparative regression.
* Added true and predicted KDMA values to the log and `input_output.json` file for comparative regression ADM.
* Added Phase 1 eval alignment targets for SoarTech

### Fixed

* Fixed KDE target samples to be between 0 and 1
* Fixed issue in alignment_utils logging (where kdma values can be a float/int rather than a list)
* Now properly hydrating the meta_info field of input_output files
* Fixed possible divide by zero during misaligned alignment
* Properly hydrate Aid list

### Deprecated

* Removed old and unused command-line interface scripts
* Removed old template files for integrating custom ADMs
* Removed CLI builder functionality
* Removed old configuration files from before Hydra

## 0.5.2

### Added

* Split out our experiment configuration for our aligned DRE ADM to specific configs for SoarTech and Adept
* Added logging for sampled KDMA target value, and estimated KDMA values in alignment_utils

### Fixed

* Fixed issue in Oracle ADM which caused an key error exception when logging probabilities

## 0.5.1

### Changed

* Updated Hybrid Kaleido ADM to optionally (on by default) use alignment_utils to support distribution based alignment
* Refactored outlines_adm to break out action parameter completion into separate functions for reuse
* Update README ADM invocation examples for the dry run evaluation (DRE)

### Added

* Added support for 'precision' in model_kwargs for outlines based adms (expecting either 'full' or 'half')
* Add option to save per scenario x alignment target unstructured outputs (useful for "eval" TA3 session types)
* Added DRE experiment configurations

### Fixed

* Fixed case in Kaleido ADM where choices weren't necessarily unique
* In outlines_adm ensure that an already tagged character can't be selected again for the TAG_CHARACTER action
* In outlines_adm ensure that already visited characters can't be selected again for assessment actions
* In outlines_adm ensure MOVE_TO specifies character ID
* In run_align_sytem CLI, don't allow unseen characters except for MOVE_TO and MOVE_TO_EVAC actions
* Typo fix for Quality of Life KDMA description

## 0.5.0

### Changed

* Updated KDMA descriptions and made the KDMA description yml file configurable
* No longer overwriting data when followup prompts are used in the Outlines ADM
* Small updates to Outlines ADM to be compatible with API updates
* Updated the oracle and comparative regression ADMs to use `AlignmentFunction` class
* Updated comparative regression ADMs justification to use the best samples reasoning

### Added

* Added incontext learning option for Outlines-based structured ADM
* Added incontext learning option for Outlines-based regression ADM
* Added alignment targets for ADEPT training scenarios for the dry run evaluation
* Added comparative regression ADM which predicts KDMA scores for all responses simultaneously, enabling comparative reasoning
* Added template option or `kdma_score_examples` for regression and comparative regression ADMs
* Added incontext learning with chain of thought reasoning for regression and comparative regression ADMs
* Added some Kaleido hybrid experiments for the ADEPT dry run scenarios
* Added Persona based ADM from UCB (based off single kdma adm)
* Added alignment targets for SoarTech scenarios for the dry run evaluation
* Added some random ADM experiments for the SoarTech dry run scenarios
* Added `intend_action` to the `ActionBasedScenarioInterface` to comply with TA3 server updates
* Added functionality in the oracle and comparative regression ADMs for aligning to KDE targets
* Added a misaligned option for the Oracle ADM using any alignment function
* Added configuration option to record timing information about `choose_action`
* Added a scenario description prompt which includes all unique structured character info
* Added a hybrid regression approach for the Outlines ADM.

### Fixed

* Fixed issue for running in batches with batch size in outlines ADMs
* Fixed character selection to use the `character_id` associated with the selected action when available, otherwise send a follow up prompt
* Restrict actions with pre-specified treatments when those supplies are not available

## 0.4.1

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
