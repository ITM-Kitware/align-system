import json

from align_system.interfaces.cli_builder import build_interfaces


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
    run_custom_action_based_system(
        **build_interfaces(add_cli_args, "My action based ALIGN System CLI",
                           supported_interfaces={'TA3ActionBased'}))


def run_custom_action_based_system(interface,
                                   model,
                                   align_to_target=False,):
    scenario = interface.start_scenario()

    if align_to_target:
        alignment_target = scenario.get_alignment_target()

    # DO ALGORITHM SETUP THINGS HERE

    current_state = scenario.get_state()
    scenario_complete = current_state.get('scenario_complete', False)

    while not scenario_complete:
        available_actions = scenario.get_available_actions()

        # DO ALGORITHM THINGS HERE

        print(json.dumps(available_actions, indent=2))

        action_to_take = available_actions[0]  # Just taking first action

        # 'APPLY_TREATMENT' actions require additional parameters to
        # be provided, i.e. the treatment type (one of the 'supplies'
        # available in the scenario, as well as the location of
        # treatment
        if action_to_take['action_type'] == 'APPLY_TREATMENT':
            action_to_take['parameters'] = {
                'treatment': current_state['supplies'][0]['type'],
                'location': 'right forearm'}

        print("** TAKING ACTION: **")
        print(json.dumps(action_to_take))

        current_state = scenario.take_action(action_to_take)
        scenario_complete = current_state.get('scenario_complete', False)


if __name__ == "__main__":
    main()
