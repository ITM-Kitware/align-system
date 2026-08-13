import json
from copy import deepcopy
import atexit
import os

from rich.logging import RichHandler
from rich.console import Console
from rich.highlighter import JSONHighlighter
from swagger_client.models import ActionTypeEnum
import hydra
from omegaconf import DictConfig, OmegaConf
from timeit import default_timer as timer

from align_system.utils import logging
from align_system.utils.hydra_utils import initialize_with_custom_references
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


def plot_mean_with_smoothing(ax, runs, color, label, window=5, no_bounds=False):
    runs = np.array(runs)
    mean = runs.mean(axis=0)
    # Compute SEM from raw runs
    std = runs.std(axis=0, ddof=1)  # sample std
    std = std / np.sqrt(runs.shape[0]) 
    
    x = np.arange(len(mean))
    
    mean_smooth = pd.Series(mean).rolling(window, min_periods=1).mean()
    
    if no_bounds:
        lower = mean_smooth - std
    else:
        lower = np.maximum(mean_smooth - std, 0)  # std from raw runs
    upper = mean_smooth + std
    ax.plot(x, mean_smooth, color=color)
    ax.fill_between(x, lower, upper, color=color, alpha=0.2)


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="action_based")
def main(cfg: DictConfig) -> None:
    cfg = initialize_with_custom_references(cfg)

    interface = cfg.interface
    adm = cfg.adm.instance

    # Using the hydra generated output directory for the run
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logfile_path = None
    if cfg.save_log:
        logfile_path = os.path.join(output_dir, "align_system.log")

    raw_logfile_path = None
    if cfg.save_raw_log:
        raw_logfile_path = os.path.join(output_dir, "raw_align_system.log")

    save_input_output_to_path = None
    if cfg.save_input_output:
        save_input_output_to_path = os.path.join(output_dir, "input_output.json")

    save_alignment_score_to_path = None
    if cfg.save_scoring_output:
        save_alignment_score_to_path = os.path.join(output_dir, "scores.json")

    save_alignment_targets_to_path = None
    if cfg.save_alignment_targets:
        save_alignment_targets_to_path = os.path.join(output_dir, "targets")
        os.mkdir(save_alignment_targets_to_path)

    save_timing_to_path = None
    if cfg.save_timing:
        save_timing_to_path = os.path.join(output_dir, "timing.json")

    # Set log level on root logger (such that child loggers respect
    # the set log level)
    root_logger = logging.getLogger()
    root_logger.setLevel(cfg.loglevel)

    def _log_closer_closure(logfile_filehandler):
        # Capture the logfile filehandler object we want to close
        # in a closure to keep a reference to it when it comes
        # time to close it
        def _close_logfile():
            logfile_filehandler.close()

        return _close_logfile

    if logfile_path is not None:
        logfile = open(logfile_path, 'w')
        # Ensure the opened logfile is closed when the program exits
        atexit.register(_log_closer_closure(logfile))

        filehandler = RichHandler(
            console=Console(file=logfile, color_system=None))
        root_logger.addHandler(filehandler)

    if raw_logfile_path is not None:
        # Using Python stdlib logging.FileHandler
        from logging import FileHandler
        filehandler = FileHandler(raw_logfile_path)

        root_logger.addHandler(filehandler)

    if cfg.get('force_determinism', False) or 'torch_random_seed' in cfg:
        import torch
        torch_seed = cfg.get('torch_random_seed', 0)
        log.info(f"Setting `torch.manual_seed` to: {torch_seed}")
        torch.manual_seed(torch_seed)

    if cfg.get('force_determinism', False) or 'torch_use_deterministic_algorithms' in cfg:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        log.info("Setting `torch_use_deterministic_algorithms` to True")
        torch.use_deterministic_algorithms(
            cfg.get('torch_use_deterministic_algorithms', True),
            warn_only=True)

    if cfg.get('force_determinism', False) or 'random_seed' in cfg:
        import random
        random_seed = cfg.get('random_seed', 0)
        log.info(f"Setting `random.seed` to: {random_seed}")
        random.seed(random_seed)

    if cfg.get('force_determinism', False) or 'numpy_random_seed' in cfg:
        import numpy as np
        numpy_random_seed = cfg.get('numpy_random_seed', 0)
        log.info(f"Setting `numpy.random.seed` to: {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if cfg.get('force_determinism', False) or 'sort_available_actions' in cfg:
        log.info("Setting `sort_available_actions` to True")
        sort_available_actions = cfg.get('sort_available_actions', True)
    else:
        sort_available_actions = False

    if 'filter_tag_character' in cfg:
        filter_tag_character = cfg.get('filter_tag_character', True)
    else:
        filter_tag_character = False

    # Defaults to True
    apply_action_filtering = cfg.get('apply_action_filtering', True)

    # HACK: need to invoke 'load_model' for ADMs that require it,
    # maybe it makes more sense to load_model in the init method for
    # those ADMs
    if hasattr(adm, 'load_model'):
        adm.load_model()

    # Capture inputs and outputs in a similar format to what's used by
    # our internal evaluation framework code
    inputs_outputs = []

    session_alignment_scores = []

    # Capture time it takes to choose each action
    action_times = { "scenarios": [] }
    def _compute_time_stats(times_s):
        n_times = len(times_s)
        total_time_s = sum(times_s)
        return {
            "n_actions_taken": n_times,
            "total_time_s": total_time_s,
            "avg_time_s": total_time_s / n_times if n_times else 0.,
            "max_time_s": max(times_s) if n_times else 0.,
            "raw_times_s": times_s
        }

    rewards, confidentiality, integrity, availability = [], [], [], []
    s = 0
    # Loop through available scenarios
    while scenario := interface.start_scenario():
        if scenario.id() == '':
            log.info("Next scenario ID is blank, assuming we're done, exiting")
            break
        log.info(f'[bold]*Scenario ID*[/bold]: {scenario.id()}')

        # Reset any decision or chat history for a new scenario
        if hasattr(adm, 'reset_history'):
            log.info("[bold]*Resetting choice history*[/bold]")
            adm.reset_history()

        if 'alignment_target' in cfg:
            alignment_target = cfg.alignment_target

            # Alignment targets specified in hydra configs require
            # some nested conversion to dict (from OmegaConf objects)
            # otherwise this can cause some downstream issues with
            # serialization
            alignment_target.kdma_values = [OmegaConf.to_container(c)
                                            if isinstance(c, DictConfig) else c
                                            for c in alignment_target.kdma_values]
        elif cfg.align_to_target:
            alignment_target = scenario.get_alignment_target()
        else:
            alignment_target = None

        log.info('[bold]*ALIGNMENT TARGET*[/bold]')
        if alignment_target is None:
            log.info('Alignment target is `None`')
        else:
            log.info(alignment_target)
            if save_alignment_targets_to_path is not None:
                alignment_target_path = os.path.join(save_alignment_targets_to_path, f"{alignment_target.id}.json")

                with open(alignment_target_path, "w") as f:
                    json.dump(alignment_target.to_dict(), f, indent=2)

        current_state = scenario.get_state()
        scenario_complete = current_state.scenario_complete

        # Tracking these to prevent getting stuck in a loop
        noop_actions = []

        sce_times_s = []

        last_scene_id = None
        sum_rewards = 0

        ep_rewards, ep_confidentiality, ep_integrity, ep_availability = [], [], [], []
        while not scenario_complete:
            current_scene_id = current_state.meta_info.scene_id
            if last_scene_id != current_scene_id:
                log.info(f"[bold]*CHANGED SCENE TO*: {current_scene_id}[/bold]",
                         extra={"markup": True})
                last_scene_id = current_scene_id

            available_actions = scenario.get_available_actions()
            # for a in available_actions:
            #     print(a.name)
            # exit()
            # available_actions = [available_actions[0]]

            if sort_available_actions:
                # Impose a fixed ordering of available actions to help
                # with determinism
                available_actions = sorted(available_actions, key=lambda a: a.unstructured)

            if filter_tag_character and apply_action_filtering:
                filtered_actions = []
                for action in available_actions:
                    if action.action_type != 'TAG_CHARACTER':
                        filtered_actions.append(action)
                available_actions = filtered_actions

            log.debug("[bold]*AVAILABLE ACTIONS*[/bold]",
                      extra={"markup": True})
            log.debug(json.dumps([a.to_dict() for a in available_actions], indent=4),
                      extra={"highlighter": JSON_HIGHLIGHTER})

            available_actions_filtered = available_actions

            if len(available_actions_filtered) == 0:
                raise RuntimeError("No available actions from filtered list!")
            elif len(available_actions_filtered) == 1:
                # log.info("** Choosing only available (filtered) action")
                action_to_take = available_actions_filtered[0]
                action_to_take.justification = "Only available (filtered) action"
            else:
                start_choose_action = timer()

                # Passing in a copy of available filtered actions to
                # prevent ADMs from modifying the originals (should
                # considering doing the same for current_state and
                # alignment_target)
                choose_action_result = adm.choose_action(
                    current_state,
                    [deepcopy(a) for a in available_actions_filtered],
                    alignment_target if cfg.align_to_target else None,
                    scenario_id=scenario.id(),
                    **cfg.adm.get('inference_kwargs', {}))

                # Handle choose action result (for backwards compatibility if no choice_info)
                if isinstance(choose_action_result, tuple):
                    action_to_take, choice_info = choose_action_result
                    if 'choice_info' in choice_info:
                        # Handle pipeline_adm
                        choice_info = choice_info['choice_info']
                else:
                    action_to_take = choose_action_result
                    choice_info = {}

                end_choose_action = timer()
                sce_times_s.append(end_choose_action - start_choose_action)
                log.debug(f"choose_action took {end_choose_action - start_choose_action} seconds")

            log.info("[bold]*ACTION BEING TAKEN*[/bold]",
                     extra={"markup": True})
            if isinstance(action_to_take, dict):
                log.info(json.dumps(action_to_take, indent=4),
                         extra={"highlighter": JSON_HIGHLIGHTER})
            else:
                log.info(json.dumps(action_to_take.to_dict(), indent=4),
                         extra={"highlighter": JSON_HIGHLIGHTER})

            # log.info("*State*: {}".format(
            #         current_state.unstructured))
            # print(s)
            # s+= 1
            action_choice_idx = None
            for i, a in enumerate(available_actions):
                if a.action_id == action_to_take.action_id:
                    action_choice_idx = i
                    break

            # Ensure that 'actions' stored in 'choice_info' are serializable
            # for info in choice_info.values():
            #     if 'action' in info:
            #         info['action'] = info['action'].to_dict()

            inputs_outputs.append({'input': {'scenario_id': scenario.id(),
                                             'alignment_target_id': alignment_target.id if cfg.align_to_target else None,
                                             'full_state': current_state.to_dict(),
                                             'state': current_state.unstructured,
                                             'choices': [a.to_dict() for a in available_actions]},
                                   'label': [{} if a.kdma_association is None else a.kdma_association for a in available_actions],
                                #    'choice_info': choice_info,
                                   'output': {'choice': action_choice_idx,
                                              'action': action_to_take.to_dict()}})
            # Save input_output after each action (gets overwritten
            # each time) so that we don't lose everything if the run
            # crashes or is interrupted.  Could treat this as we do
            # the logfile and open the file handle once and close
            # `atexit` and write each line as it's generated (and make
            # it a .jsonl file; would need to remove the indent=2)
            if save_input_output_to_path is not None:
                with open(save_input_output_to_path, 'w') as f:
                    json.dump(inputs_outputs, f, indent=2)

            last_state = current_state
            try:
                if hasattr(action_to_take, "intent_action") and action_to_take.intent_action:
                    current_state = scenario.intend_action(action_to_take)
                else:
                    current_state = scenario.take_action(action_to_take)
            except Exception as e:
                log.info(action_to_take)
                raise e

            sum_rewards += scenario.reward
            log.info("*Reward*: {}".format(scenario.reward))
            log.info("*CIA Scores*: {}".format(scenario.cia_scores))

            ep_rewards.append(scenario.reward)
            ep_confidentiality.append(scenario.cia_scores[0])
            ep_integrity.append(scenario.cia_scores[1])
            ep_availability.append(scenario.cia_scores[2])

            # Check that the scenario state has really changed
            # Want to restrict actions that have already been taken that
            # didn't change the state
            _tmp_current_state = deepcopy(current_state)
            _tmp_current_state.elapsed_time = last_state.elapsed_time
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
                log.info("*Final state unstructured*: {}".format(
                    current_state.unstructured))

                if cfg.get('save_last_unstructured_state_per_scenario', False):
                    if alignment_target is None:
                        scenario_alignment_target = scenario.get_alignment_target()

                        if scenario_alignment_target is not None:
                            alignment_target_id = scenario_alignment_target.id
                        else:
                            alignment_target_id = None
                    else:
                        alignment_target_id = alignment_target.id

                    final_scenario_state_output_path = os.path.join(
                        output_dir, "{}.{}.final_state_unstructured.json".format(
                            scenario.id(), alignment_target_id))
                    with open(final_scenario_state_output_path, "w") as f:
                        print(current_state.unstructured, file=f)

        log.info("*Sum of rewards in episode*: {}".format(sum_rewards))
        s += sum_rewards
        rewards.append(np.array(ep_rewards))
        confidentiality.append(np.array(ep_confidentiality))
        integrity.append(np.array(ep_integrity))
        availability.append(np.array(ep_availability))

        if save_timing_to_path is not None:
            action_times["scenarios"].append(_compute_time_stats(sce_times_s))

        if alignment_target is not None:
            try:
                session_alignment = interface.get_session_alignment(
                    alignment_target)
            except Exception:
                # Could be more specific about what kind of exceptions
                # to expect here
                session_alignment = None

            if session_alignment is None:
                log.info("Couldn't get session alignment from interface")
            else:
                session_alignment_scores.append(session_alignment)

                if isinstance(session_alignment, dict):
                    session_alignment_dict = session_alignment
                else:
                    session_alignment_dict = session_alignment.to_dict()

                log.info("[bold]*TA1 Alignment Score*[/bold]",
                         extra={"markup": True})
                log.info(json.dumps(session_alignment_dict, indent=4),
                         extra={"highlighter": JSON_HIGHLIGHTER})

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    plot_mean_with_smoothing(axs[0], confidentiality, "purple", "Confidentiality")
    plot_mean_with_smoothing(axs[1], integrity, "purple", "Integrity")
    plot_mean_with_smoothing(axs[2], availability, "purple", "Availability")
    # plot_mean_with_smoothing(axs[3], resiliences, "blue", "Resilience")
    plot_mean_with_smoothing(axs[3], rewards, "green", "Reward", no_bounds=True)

    axs[0].set_title("Confidentiality Drop")
    axs[0].invert_yaxis()
    axs[1].set_title("Integrity Drop")
    axs[1].invert_yaxis()
    axs[2].set_title("Availability Drop")
    axs[2].invert_yaxis()
    # axs[3].set_title("Resilience Drop")
    # axs[3].invert_yaxis()
    axs[3].set_title("CAGE Reward")
    fig.text(0.5, 0.04, "Game Step", ha="center")
    fig.text(0.04, 0.5, "Score", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, "Metrics.png"))

    print('Average return: {}'.format(s / 30))
    
    if save_timing_to_path is not None:
        all_times = []
        for sce in action_times["scenarios"]:
            all_times.extend(sce["raw_times_s"])

        action_times.update(_compute_time_stats(all_times))

        with open(save_timing_to_path, 'w') as f:
            json.dump(action_times, f, indent=2)

    if len(session_alignment_scores) > 0:
        if save_alignment_score_to_path is not None:
            with open(save_alignment_score_to_path, 'w') as f:
                json.dump([(s if isinstance(s, dict) else s.to_dict())
                           for s in session_alignment_scores], f, indent=2)


if __name__ == "__main__":
    main()
