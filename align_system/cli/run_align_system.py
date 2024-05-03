import sys
import json
import yaml
from copy import deepcopy
import atexit
from itertools import chain

from rich.logging import RichHandler
from rich.console import Console
from rich.highlighter import JSONHighlighter
from swagger_client.models import AlignmentTarget, ActionTypeEnum, InjuryStatusEnum

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
    parser.add_argument('--logfile-path',
                        type=str,
                        default=None,
                        help="Also write log output to the specified file")
    parser.add_argument('--save-input-output-to-path',
                        type=str,
                        default=None,
                        help="Save system inputs and outputs to a file")
    parser.add_argument('--save-alignment-score-to-path',
                        type=str,
                        default=None,
                        help="Save alignment score output to a file")


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
            add_cli_args, "ALIGN System CLI",
            supported_interfaces={'TA3ActionBased'}))


def run_action_based_chat_system(interface,
                                 adm_config,
                                 align_to_target,
                                 loglevel="INFO",
                                 logfile_path=None,
                                 save_input_output_to_path=None,
                                 save_alignment_score_to_path=None):
    # Set log level on root logger (such that child loggers respect
    # the set log level)
    root_logger = logging.getLogger()
    root_logger.setLevel(loglevel)

    if logfile_path is not None:
        logfile = open(logfile_path, 'w')
        # Ensure the opened logfile is closed when the program exits
        atexit.register(lambda: logfile.close())

        filehandler = RichHandler(
            console=Console(file=logfile, color_system=None))
        root_logger.addHandler(filehandler)

    with open(adm_config, 'r') as f:
        config = yaml.safe_load(f)

    adm_config = config['adm']
    adm_name = adm_config['name']
    adm_init_kwargs = adm_config.get('init_kwargs', {})
    adm_inference_kwargs = adm_config.get('inference_kwargs', {})
    adm_class = REGISTERED_ADMS.get(adm_name)

    if adm_class is None:
        raise RuntimeError("'adm' not found in REGISTERED_ADMS: {}".format(
            list(REGISTERED_ADMS.keys())))

    # TODO: Check that the selected ADM implements the expected
    # abstract with respect to the selected "interface"
    # (i.e. TA3ActionBased, vs. TA1)
    adm = adm_class(**adm_init_kwargs)

    # HACK: need to invoke 'load_model' for ADMs that require it,
    # maybe it makes more sense to load_model in the init method for
    # those ADMs
    if hasattr(adm, 'load_model'):
        adm.load_model()

    # Capture inputs and outputs in a similar format to what's used by
    # our internal evaluation framework code
    inputs_outputs = []

    session_alignment_scores = []

    # Loop through available scenarios
    while scenario := interface.start_scenario():
        if scenario.id() == '':
            log.info("Next scenario ID is blank, assuming we're done, exiting")
            break

        if 'alignment_target_override' in config:
            alignment_target = AlignmentTarget(
                **config['alignment_target_override'])
        elif align_to_target:
            alignment_target = scenario.get_alignment_target()
        else:
            alignment_target = None

        current_state = scenario.get_state()
        scenario_complete = current_state.scenario_complete

        # Tracking these to prevent getting stuck in a loop
        noop_actions = []

        while not scenario_complete:
            available_actions = scenario.get_available_actions()

            log.debug("[bold]*AVAILABLE ACTIONS*[/bold]",
                      extra={"markup": True})
            log.debug(json.dumps([a.to_dict() for a in available_actions], indent=4),
                      extra={"highlighter": JSON_HIGHLIGHTER})

            available_actions_filtered = []
            for a in available_actions:
                if len(current_state.characters) == 0:
                    # Restrict actions that require a character when
                    # no characters exist
                    if a.action_type in {ActionTypeEnum.APPLY_TREATMENT,
                                         ActionTypeEnum.CHECK_ALL_VITALS,
                                         ActionTypeEnum.CHECK_PULSE,
                                         ActionTypeEnum.CHECK_RESPIRATION,
                                         ActionTypeEnum.MOVE_TO_EVAC,
                                         ActionTypeEnum.TAG_CHARACTER}:
                        log.debug("No characters in current state, not "
                                  "allowing {} action".format(a.action_type))
                        continue

                if a.action_type == ActionTypeEnum.TAG_CHARACTER:
                    # Don't let ADM choose to tag a character unless there are
                    # still untagged characters
                    untagged_characters = [c for c in current_state.characters
                                           if c.tag is None]
                    if len(untagged_characters) == 0:
                        log.debug("No untagged characters remaining, not "
                                  "allowing {} action".format(ActionTypeEnum.TAG_CHARACTER))
                        continue

                unvisited_characters = [c for c in current_state.characters
                                        if c.visited is None or not c.visited]
                if a.action_type in {ActionTypeEnum.CHECK_ALL_VITALS,
                                     ActionTypeEnum.CHECK_PULSE,
                                     ActionTypeEnum.CHECK_RESPIRATION}:
                    if len(unvisited_characters) == 0:
                        log.debug("No unvisited characters remaining, not "
                                  "allowing {} action".format(a.action_type))
                        continue

                if a.action_type == ActionTypeEnum.APPLY_TREATMENT:
                    untreated_injuries = [i for i in chain(*(c.injuries for c in current_state.characters)) if i.status != InjuryStatusEnum.TREATED]

                    if len(untreated_injuries) == 0:
                        log.debug("No untreated injuries remaining, not "
                                  "allowing {} action".format(a.action_type))
                        continue

                    if a.character_id is not None:
                        matching_characters = [c for c in current_state.characters
                                               if c.id == a.character_id]
                        assert len(matching_characters) == 1
                        treated_injuries_for_character = [i for i in matching_characters[0].injuries if i.status == InjuryStatusEnum.TREATED]

                        if len(treated_injuries_for_character) > 0:
                            log.debug("Treated at least one injury for {}, not "
                                      "allowing {} action".format(a.character_id, a.action_type))
                            continue

                is_a_noop_action = False
                for noop_action in noop_actions:
                    if a == noop_action:
                        is_a_noop_action = True

                    # HACK: In some cases the ADM can get stuck
                    # attempting to use the generic APPLY_TREATMENT
                    # action over and over to no affect
                    if noop_action.action_type == ActionTypeEnum.APPLY_TREATMENT:
                        _tmp_noop_action = deepcopy(noop_action)

                        _tmp_noop_action.parameters = None
                        _tmp_noop_action.character_id = None

                        if a == _tmp_noop_action:
                            is_a_noop_action = True
                            log.debug("Handled case where ADM might be stuck "
                                      "applying treatment over and over to no "
                                      "effect, not allowing {} action".format(a.action_type))

                if is_a_noop_action:
                    log.debug("Already took this action and there was no "
                              "change in the scenario state, not allowing "
                              "{} action".format(a.action_type))
                    continue

                available_actions_filtered.append(a)

            if len(available_actions_filtered) == 0:
                raise RuntimeError("No available actions from filtered list!")
            elif len(available_actions_filtered) == 1:
                log.info("** Choosing only available (filtered) action")
                action_to_take = available_actions_filtered[0]
            else:
                action_to_take = adm.choose_action(
                    current_state,
                    available_actions_filtered,
                    alignment_target if align_to_target else None,
                    **adm_inference_kwargs)

            log.debug("[bold]*ACTION BEING TAKEN*[/bold]",
                      extra={"markup": True})
            if isinstance(action_to_take, dict):
                log.debug(json.dumps(action_to_take, indent=4),
                          extra={"highlighter": JSON_HIGHLIGHTER})
            else:
                log.debug(json.dumps(action_to_take.to_dict(), indent=4),
                          extra={"highlighter": JSON_HIGHLIGHTER})

            action_choice_idx = None
            for i, a in enumerate(available_actions):
                if a.action_id == action_to_take.action_id:
                    action_choice_idx = i
                    break

            inputs_outputs.append({'input': {'scenario_id': scenario.id(),
                                             'full_state': current_state.to_dict(),
                                             'state': current_state.unstructured,
                                             'choices': [a.to_dict() for a in available_actions]},
                                   'label': [{} if a.kdma_association is None else a.kdma_association for a in available_actions],
                                   'output': {'choice': action_choice_idx,
                                              'action': action_to_take.to_dict()}})

            last_state = current_state
            current_state = scenario.take_action(action_to_take)
            log.debug("Elapsed time for current state: {}".format(current_state.elapsed_time))

            # Check that the scenario state has really changed
            # Want to restrict actions that have already been taken that
            # didn't change the state
            _tmp_current_state = deepcopy(current_state)
            _tmp_current_state.elapsed_time = last_state.elapsed_time
            _tmp_current_state.supplies = last_state.supplies

            state_has_changed = (_tmp_current_state != last_state)
            if state_has_changed:
                noop_actions = []
            else:
                # Strip out the justification string (provided by our
                # ADMs) from no-op actions so that it can be compared
                # to the original actions
                _tmp_action_to_take = deepcopy(action_to_take)
                _tmp_action_to_take.justification = None
                noop_actions.append(_tmp_action_to_take)

            scenario_complete = current_state.scenario_complete

            if scenario_complete:
                log.info("Final state unstructured: {}".format(
                    current_state.unstructured))

        if alignment_target is not None:
            session_alignment = interface.get_session_alignment(
                alignment_target.id)

            if session_alignment is None:
                log.info("Couldn't get session alignment from interface")
            else:
                session_alignment_scores.append(session_alignment)

                log.info("[bold]*TA1 Alignment Score*[/bold]",
                         extra={"markup": True})
                log.info(json.dumps(session_alignment.to_dict(), indent=4),
                         extra={"highlighter": JSON_HIGHLIGHTER})

    if save_input_output_to_path is not None:
        with open(save_input_output_to_path, 'w') as f:
            json.dump(inputs_outputs, f, indent=2)

    if len(session_alignment_scores) > 0:
        if save_alignment_score_to_path is not None:
            with open(save_alignment_score_to_path, 'w') as f:
                json.dump([s.to_dict() for s in session_alignment_scores], f, indent=2)


if __name__ == "__main__":
    main()
