import sys
import json
import yaml

from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.interfaces.cli_builder import build_interfaces
from align_system.algorithms import REGISTERED_ADMS


log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


def add_cli_args(parser):
    # Using argparse to add our system CLI specific arguments.  Can
    # modify or add your own custom CLI arguments here
    parser.add_argument('-c', '--adm-config',
                        type=str,
                        required=True,
                        help="Path to ADM config YAML")
    parser.add_argument('-t', '--align-to-target',
                        action='store_true',
                        default=False,
                        help="Align algorithm to target KDMAs")
    parser.add_argument('-l', '--loglevel',
                        type=str,
                        default='INFO')


def main():
    # The `build_interfaces` call here adds all interfaces as
    # subparsers to your CLI.  (Can specify what interfaces you
    # support explicitly with the optional `supported_interfaces`
    # argument (as a set))
    # The `build_interfaces` call also instantiates an interface
    # object based on the selected interface and interface arguments
    # provided at the command line and passes them to your run
    # function (`run_custom_system` in this case)
    log.debug(f"[bright_black]CMD: {' '.join(sys.argv)}[/bright_black]",
              extra={'markup': True, 'highlighter': None})
    run_action_based_chat_system(
        **build_interfaces(
            add_cli_args, "ALIGN Action Based System CLI - Chat Model",
            supported_interfaces={'TA3ActionBased'}))


def run_action_based_chat_system(interface,
                                 adm_config,
                                 align_to_target,
                                 loglevel="INFO"):
    # Set log level on root logger (such that child loggers respect
    # the set log level)
    logging.getLogger().setLevel(loglevel)

    with open(adm_config, 'r') as f:
        config = yaml.safe_load(f)

    adm_config = config['adm']
    adm_name = adm_config['name']
    adm_init_kwargs = adm_config.get('init_kwargs', {})
    adm_inference_kwargs = adm_config.get('inference_kwargs', {})
    adm_class = REGISTERED_ADMS.get(adm_name)

    if adm_class is None:
        raise RuntimeError("'adm' not found in REGISTERED_ADMS: {}".format(
            REGISTERED_ADMS.key()))

    # TODO: Check that the selected ADM implements the expected
    # abstract with respect to the selected "interface"
    # (i.e. TA3ActionBased, vs. TA1)
    adm = adm_class(**adm_init_kwargs)

    scenario = interface.start_scenario()

    if align_to_target:
        alignment_target = scenario.get_alignment_target()
    else:
        alignment_target = None

    current_state = scenario.get_state()
    scenario_complete = current_state.scenario_complete

    while not scenario_complete:
        available_actions = scenario.get_available_actions()

        log.debug("[bold]*AVAILABLE ACTIONS*[/bold]",
                  extra={"markup": True})
        log.debug(json.dumps([a.to_dict() for a in available_actions], indent=4),
                  extra={"highlighter": JSON_HIGHLIGHTER})

        untagged_characters = [c for c in current_state.characters if c.tag is None]

        # Don't let ADM choose to tag a character unless there are
        # still untagged characters
        available_actions_filtered =\
            [a for a in available_actions
             if a.action_type != 'TAG_CHARACTER'
             or (a.action_type == 'TAG_CHARACTER'
                 and len(untagged_characters) > 0)]

        if len(available_actions_filtered) == 0:
            raise RuntimeError("No available actions from filtered list!")
        elif len(available_actions_filtered) == 1:
            log.info("** Choosing only available (filtered) action")
            action_to_take = available_actions_filtered[0]
        else:
            action_to_take = adm.choose_action(
                current_state,
                available_actions_filtered,
                alignment_target,
                **adm_inference_kwargs)

        log.debug("[bold]*ACTION BEING TAKEN*[/bold]",
                  extra={"markup": True})
        if isinstance(action_to_take, dict):
            log.debug(json.dumps(action_to_take, indent=4),
                      extra={"highlighter": JSON_HIGHLIGHTER})
        else:
            log.debug(json.dumps(action_to_take.to_dict(), indent=4),
                      extra={"highlighter": JSON_HIGHLIGHTER})

        current_state = scenario.take_action(action_to_take)

        scenario_complete = current_state.scenario_complete


if __name__ == "__main__":
    main()
