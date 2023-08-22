# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

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
