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

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()
from collections import deque


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="action_based")
def main(cfg: DictConfig) -> None:
    choice_info = None
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
    predictions = [0, 0 , 0]
    # Loop through available scenarios
    while scenario := interface.start_scenario():
        collected_observations = deque(maxlen = 2)
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

        while not scenario_complete:
            collected_observations.append(current_state.unstructured)
            current_scene_id = current_state.meta_info.scene_id
            if last_scene_id != current_scene_id:
                log.info(f"[bold]*CHANGED SCENE TO*: {current_scene_id}[/bold]",
                         extra={"markup": True})
                last_scene_id = current_scene_id

            available_actions = scenario.get_available_actions()
            action_to_take = available_actions[0]
            
            if save_input_output_to_path is not None:
                with open(save_input_output_to_path, 'w') as f:
                    json.dump(inputs_outputs, f, indent=2)

            try:
                if hasattr(action_to_take, "intent_action") and action_to_take.intent_action:
                    current_state = scenario.intend_action(action_to_take)
                else:
                    current_state = scenario.take_action(action_to_take)
            except Exception as e:
                log.info(action_to_take)
                raise e

            scenario_complete = current_state.scenario_complete

        
        observation = ''
        for i in range(len(collected_observations)):
            observation += 'Observation {0}:'.format(i) + '\n'
            observation += collected_observations[i] + '\n'

        # print(observation)
        log.info(observation)
        current_state.unstructured = observation
        available_actions = scenario.get_available_Red_actions()

        choose_action_result = adm.choose_action(
                current_state,
                [deepcopy(a) for a in available_actions],
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


        log.info("[bold]*ACTION BEING TAKEN*[/bold]",
                    extra={"markup": True})
        
        if isinstance(action_to_take, dict):
                log.info(json.dumps(action_to_take, indent=4),
                         extra={"highlighter": JSON_HIGHLIGHTER})
        else:
            log.info(json.dumps(action_to_take.to_dict(), indent=4),
                        extra={"highlighter": JSON_HIGHLIGHTER})
            
        if action_to_take.name == 'Sleep':
            predictions[0] += 1
        elif action_to_take.name == 'Meander':
            predictions[1] += 1
        elif action_to_take.name == 'B-line':   
            predictions[2] += 1

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
    log.info('Red agents classification results:')
    log.info(predictions)

if __name__ == "__main__":
    main()
