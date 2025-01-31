# Integration Tests

For integration testing, we run complete ADM configs (with
deterministic options enabled) and compare against expected ouptuts.
This allows us to verify when and how our ADM outputs change as a
result of dependency updates and software changes.

## Running an integration test

Integration test configs are stored in the
`align_system/configs/experiment/integration_tests` subdirectory, and
the intention is that each of these captures a complete ADM
configuration (including an alignment target for aligned ADMs).  Each
of these integration test configs should already have expected outputs
generated for them in `tests/data/expected_outputs/integration_tests`
under subdirectories named after each config.

A dedicated integration test driver script
(`tests/run_integration_test.py`) is responsible for running the ADM
configuration and comparing the run outputs against the expected
outputs.  If they are different, a diff will be shown indicating the
differences.  This script exits with a status of `0` if there are no
differences, and a status of `1` otherwise.

To run the integration test for one of these configs:
```
python tests/run_integration_test.py --show-experiment-stdout integration_tests/random
```

## Adding a new integration test

To add a new integration test, a new configuration file should be
added in the `align_system/configs/experiment/integration_tests`
subdirectory.  This config should use the `input_output_file`
interface (and specify the path to the input-output file to be used),
as well as specify an alignment target in the case of an aligned ADM.

For example, here's the contents of the [random](align_system/configs/experiment/integration_tests/random.yaml) configuration file:
```yaml
# @package _global_
defaults:
  - override /adm: random
  - override /interface: input_output_file

interface:
  input_output_filepath: ${test_data_dir}/adept-mj1-train-subset.json

force_determinism: true
```

Note how we're using the `${test_data_dir}` variable, in the
`run_integration_test.py` script this gets replaced with the path to
the test data directory (i.e. `tests/data`).

Also, we're using `force_determinism: true` to ensure reproducability
(the entire premise of these integration tests is based on
deterministic runs).

Here's the first few lines from one of the aligned ADM configurations
for integration testing
[comp_reg_icl_adept_1.yaml](align_system/configs/experiment/integration_tests/comp_reg_icl_adept_1.yaml):
```yaml
# @package _global_
defaults:
  - /alignment_target: "ADEPT-DryRun-Moral judgement-0.2"
  - override /adm: outlines_regression_aligned_comparative/incontext_phase1
  - override /interface: input_output_file

interface:
  input_output_filepath: ${test_data_dir}/adept-mj1-train-subset.json

...
```

Note how we're explicitly setting the alignment target for the test
(rather than at runtime as we typically do for inference)

### Generating expected output files for your test

The `run_integration_tests.py` script is able to generated expected
outputs in the right place by including the `-r`
(`--replace-expected-outputs`) flag in the invocation:
```
python tests/run_integration_test.py --replace-expected-outputs --show-experiment-stdout integration_tests/random
```

Note that for a newly added test, you'll want to commit the generated
expected output files.  A `.gitignore` file is generated along with
the expected outputs to ensure that only the files to be checked with
subsequent runs are committed.
