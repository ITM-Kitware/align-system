# Creating a Custom System Script

The `align-system` code provides some building blocks for creating your own system-level CLI script.  Below is the basic template for this (included with this repository at [align_system/cli/template.py](/align_system/cli/template.py)):

```python
import json

from align_system.interfaces.cli_builder import build_interfaces
from align_system.utils.enums import ProbeType
from align_system.interfaces.abstracts import (
    ScenarioInterfaceWithAlignment,
    ProbeInterfaceWithAlignment)


def add_cli_args(parser):
    # Using argparse to add our system CLI specific arguments.  Can
    # modify or add your own custom CLI arguments here
    parser.add_argument('-m', '--model',
                        type=str,
                        help="Example command-line argument")
    parser.add_argument('-t', '--align-to-target',
                        action='store_true',
                        default=False,
                        help="Align algorithm to target KDMAs")


def main():
    # The `build_interfaces` call here adds all interfaces as
    # subparsers to your CLI.  (Can specify what interfaces you
    # support explicitly with the optional `supported_interfaces`
    # argument (as a set))
    # The `build_interfaces` call also instantiates an interface
    # object based on the selected interface and interface arguments
    # provided at the command line and passes them to your run
    # function (`run_custom_system` in this case)
    run_custom_system(**build_interfaces(add_cli_args, "My ALIGN System CLI"))


def run_custom_system(interface,
                      model,
                      align_to_target=False,):
    scenario = interface.start_scenario()
    scenario_dict = scenario.to_dict()

    if align_to_target:
        alignment_target = scenario.get_alignment_target()
        alignment_target_dict = alignment_target.dict()

    # DO ALGORITHM SETUP THINGS HERE

    for probe in scenario.iterate_probes():
        print(probe.pretty_print_str())
        print()

        probe_dict = probe.to_dict()

        # DO ALGORITHM THINGS HERE

        # Placeholder value:
        justification_str = "This seems like the correct answer"

        if probe_dict['type'] == ProbeType.MultipleChoice.value:
            # Placeholder value:
            selected_choice_id = probe_dict['options'][0]['id']  # First option

            probe_response = {'justification': justification_str,
                              'choice': selected_choice_id}
        else:
            probe_response = {'justification': justification_str}

        print(json.dumps(probe_response, indent=2))
        print()

        probe.respond(probe_response)

        # Get KDMA Alignment scores for probe if the interface supports it
        if isinstance(probe, ProbeInterfaceWithAlignment):
            probe_alignment_results = probe.get_alignment_results()
            print(json.dumps(probe_alignment_results, indent=2))
            print()

    # Get KDMA Alignment scores for scenario if the interface supports it
    if isinstance(scenario, ScenarioInterfaceWithAlignment):
        scenario_alignment_results = scenario.get_alignment_results()
        print(json.dumps(scenario_alignment_results, indent=2))
        print()


if __name__ == "__main__":
    main()
```


Note that this template is runable as-is (it simply responds with placeholder values).

## Other useful components

The `align-system` code also has some other potentially useful components for your custom system script, usage examples of several of these can be found in the main CLI [align_system/cli/run_align_system.py](/align_system/cli/run_align_system.py).

### Prompt engineering

We have a generic prompt generation function called `prepare_prompt`:

```python
from align_system.prompt_engineering.common import prepare_prompt

scenario_unstructured = "Initial setting: your in an desert environment with 2 other squad members\n"
mission_unstructured = "Treat injured squad members"
state_unstructured = "Squad member A is hemorrhaging, squad member B is unconscious"
probe_prompt = "Who do you treat first?"
casualties = [{"unstructured": "Squad member A"},
              {"unstructured": "Squad member B"}]
probe_options = [{"value": "Treat squad member A"},
                 {"value": "Treat squad member B"}]
align_to_target = True
alignment_target = {"kdma_values": [{"kdma": "mission", "value": 1},
                                    {"kdma": "denial", "value": 3}]}

prompt = prepare_prompt(
    scenario_unstructured,
    mission_unstructured,
    state_unstructured,
    probe_prompt,
    casualties,
    options=probe_options,
    alignment_target=alignment_target if align_to_target else None
)

print(prompt)
```

Which produces:
```
Instruction: Use medical triage knowledge to answer the following question with emphasis on mission, and heavy emphasis on denial.
Initial setting: your in an desert environment with 2 other squad members
Squad member A
Squad member B

Mission for the task is : Treat injured squad members
Squad member A is hemorrhaging, squad member B is unconscious
Who do you treat first? Choose from the following:
A: Treat squad member A
B: Treat squad member B

Response:
```

### Forced choice selection from raw text

We provide a few options for forcing a choice from a text string and a set of options:

```python
from align_system.similarity_measures import build_force_choice_func

similarity_measure = "heuristic"
force_choice_func = build_force_choice_func(similarity_measure)

raw_text = "Squad member A should be treated first"
options = ["Squad member A", "Squad member B"]

selected_option_index, selected_option = force_choice_func(raw_text, options)

print(selected_option)
```

Which produces:
```
Squad member A
```
