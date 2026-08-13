import json
from copy import deepcopy
import atexit
import os
import inspect
from rich.logging import RichHandler
from rich.console import Console
from rich.highlighter import JSONHighlighter
from swagger_client.models import ActionTypeEnum
import hydra
from omegaconf import DictConfig, OmegaConf
from timeit import default_timer as timer

from align_system.utils import logging
from align_system.utils.hydra_utils import initialize_with_custom_references
from cyborg.Agents.HMARLAgents.submission import Submission
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


def action_space_change(action_signature, action_space: dict) -> int:
        assert type(action_space) is dict, \
            f"Wrapper required a dictionary action space. " \
            f"Please check that the wrappers below the ReduceActionSpaceWrapper return the action space as a dict "
        possible_actions = []
        temp = {}
        params = ['action']
        # for action in action_space['action']:
        for i, action in enumerate(action_space['action']):
            if action not in action_signature:
                action_signature[action] = inspect.signature(action).parameters
            param_dict = {}
            param_list = [{}]
            for p in action_signature[action]:
                if p == 'priority':
                    continue
                temp[p] = []
                if p not in params:
                    params.append(p)

                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            p_dict[p] = key
                            new_param_list.append({key: value for key, value in p_dict.items()})
                    param_list = new_param_list
            for p_dict in param_list:
                possible_actions.append(action(**p_dict))
        return possible_actions

@hydra.main(version_base=None,
            config_path="../configs",
            config_name="action_based")
def main(cfg: DictConfig) -> None:    
    cfg = initialize_with_custom_references(cfg)

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))

    interface = cfg.interface

    algorithm = 'LLM + RL'

    if algorithm == 'LLM + RL':
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


    if algorithm == 'LLM + RL' and hasattr(adm, 'load_model'):
        adm.load_model()

    rewards, confidentiality, integrity, availability, resiliences = [], [], [], [], []
    returns = []

    while scenario := interface.start_scenario():
        if scenario.id() == '':
            log.info("Next scenario ID is blank, assuming we're done, exiting")
            break

        current_state = scenario.get_state()
        scenario_complete = current_state.scenario_complete
        sum_rewards = 0
        ep_rewards, ep_confidentiality, ep_integrity, ep_availability, ep_resiliences = [], [], [], [], []
        s = 0
        counter = 0
        analyze = 0
        while not scenario_complete:

            actions = {
                agent_name: Submission.AGENTS[agent_name].get_action(
                    scenario.observation[agent_name], interface.wrapped_cyborg.action_space(agent_name[:12])
                )
                for agent_name in scenario.observation
            }
            if s % 2 == 1:
                print(f"Step {s//2}")
                available_actions = scenario.get_available_meta_actions()
                sum_rewards += scenario.reward
                print(actions)

                choose_action_result = adm.choose_action(
                    current_state,
                    [deepcopy(a) for a in available_actions],
                    alignment_target if cfg.align_to_target else None,
                    scenario_id=scenario.id(),
                    **cfg.adm.get('inference_kwargs', {}))
                log.info(current_state.unstructured)
                action_to_take, _ = choose_action_result

                log.info(json.dumps(action_to_take['blue_agent_0'].to_dict(), indent=4), extra={"highlighter": JSON_HIGHLIGHTER})
                log.info(json.dumps(action_to_take['blue_agent_1'].to_dict(), indent=4), extra={"highlighter": JSON_HIGHLIGHTER})
                log.info(json.dumps(action_to_take['blue_agent_2'].to_dict(), indent=4), extra={"highlighter": JSON_HIGHLIGHTER})
                log.info(json.dumps(action_to_take['blue_agent_3'].to_dict(), indent=4), extra={"highlighter": JSON_HIGHLIGHTER})
                log.info(json.dumps(action_to_take['blue_agent_4'].to_dict(), indent=4), extra={"highlighter": JSON_HIGHLIGHTER})

                for agent in scenario.agents:
                    # print(scenario.info)
                    if scenario.info[agent + '_master']['success'].name != 'IN_PROGRESS':
                        # decision = 0
                        decision = 0 if action_to_take[agent].name == 'Investigate' else 1 
                        if decision != actions[agent+'_master']:
                            counter += 1
                            print('Wrong!!!!')
                        if actions[agent+'_master'] == 1:
                            analyze += 1
            s += 1      

            scenario.step(actions)
            scenario_complete = current_state.scenario_complete

        returns.append(sum_rewards)
        
        print("*Sum of rewards in episode*: {0}".format(sum_rewards))
        rewards.append(np.array(ep_rewards))
        

    print('Average reward is: {0} with a standard deviation of {1}'.format(np.mean(returns), np.std(returns)))
    print('Number of misalignments: {0}'.format(counter))
if __name__ == "__main__":
    main()
