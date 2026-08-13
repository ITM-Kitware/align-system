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
from CybORG.Agents.SimpleAgents.PPOAgent import PPOAgent
from align_system.interfaces.cage2_service import CAGEState
from align_system.utils import logging
from align_system.utils.hydra_utils import initialize_with_custom_references
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()
from collections import deque
import numpy as np
import pandas as pd
import random
import torch

from CybORG.Agents.SimpleAgents.SleepAgent import SleepAgent
from CybORG.Agents.SimpleAgents.Meander_Resilience import RedMeanderAgent_Resilience
from CybORG.Agents.SimpleAgents.B_line_resilience import B_lineAgent_Resilience


def save_rng_state():
    return {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state):
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])

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

    algorithm = 'Robust RL'
    algorithm = 'Oracle'
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

    # if cfg.get('force_determinism', False) or 'torch_random_seed' in cfg:
    #     import torch
    #     torch_seed = cfg.get('torch_random_seed', 0)
    #     log.info(f"Setting `torch.manual_seed` to: {torch_seed}")
    #     torch.manual_seed(torch_seed)

    # if cfg.get('force_determinism', False) or 'torch_use_deterministic_algorithms' in cfg:
    #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #     import torch
    #     log.info("Setting `torch_use_deterministic_algorithms` to True")
    #     torch.use_deterministic_algorithms(
    #         cfg.get('torch_use_deterministic_algorithms', True),
    #         warn_only=True)

    # if cfg.get('force_determinism', False) or 'random_seed' in cfg:
    #     import random
    #     random_seed = cfg.get('random_seed', 0)
    #     log.info(f"Setting `random.seed` to: {random_seed}")
    #     random.seed(random_seed)

    # if cfg.get('force_determinism', False) or 'numpy_random_seed' in cfg:
    #     import numpy as np
    #     numpy_random_seed = cfg.get('numpy_random_seed', 0)
    #     log.info(f"Setting `numpy.random.seed` to: {numpy_random_seed}")
    #     np.random.seed(numpy_random_seed)


    if algorithm == 'LLM + RL' and hasattr(adm, 'load_model'):
        adm.load_model()

    rng_state = save_rng_state()
    blue_sleep = PPOAgent(model_file = "./data_models/PPO_base_0_seed1_SleepAgent")
    blue_meander = PPOAgent(model_file = "./data_models/PPO_base_0_seed5_RedMeanderAgent_Resilience")
    blue_b_line = PPOAgent(model_file = "./data_models/PPO_base_0_seed5_B_lineAgent_Resilience")
    blue_robust = PPOAgent(model_file = "./data_models/PPO_base_0_seed5_B_lineAgent_Resilience_C")
    restore_rng_state(rng_state)

    rewards, confidentiality, integrity, availability, resiliences = [], [], [], [], []
    returns = []
    # predictions = [0, 0 , 0]

    while scenario := interface.start_scenario():
        if scenario.id() == '':
            log.info("Next scenario ID is blank, assuming we're done, exiting")
            break

        # rng_state = save_rng_state()
        if algorithm == 'Oracle':
        # blue_agent = PPOAgent(model_file = "./data_models/PPO_base_0_seed1_SleepAgent", seed = None)
            if interface.red_agent.__name__ == 'SleepAgent':
                blue_agent = blue_sleep
            elif interface.red_agent.__name__ == 'RedMeanderAgent_Resilience':
                blue_agent = blue_meander
            else:
                blue_agent = blue_b_line
        elif algorithm == 'LLM + RL':
            # Reset any decision or chat history for a new scenario
            if hasattr(adm, 'reset_history'):
                # log.info("[bold]*Resetting choice history*[/bold]")
                adm.reset_history()
            blue_agent = SleepAgent()
            collected_observations = deque(maxlen = 2)
        else:
            blue_agent = blue_robust
            # blue_agent = blue_meander
            # blue_agent = blue_b_line
        # restore_rng_state(rng_state)

        possible_action = action_space_change({}, interface.wrapped_cyborg.get_action_space(agent = 'Blue'))

        current_state = scenario.get_state()
        scenario_complete = current_state.scenario_complete

        sum_rewards = 0
        s = 0
        ep_rewards, ep_confidentiality, ep_integrity, ep_availability, ep_resiliences = [], [], [], [], []
        while not scenario_complete:
            if algorithm == 'LLM + RL' and s < 4:
                collected_observations.append(current_state.unstructured)
            elif algorithm == 'LLM + RL' and s == 4:
                observation = ''
                for i in collected_observations:
                    observation += i + '\n'
                log.info(observation)
                obs = CAGEState(observation, 'A', 2)
                
                # print(current_state.unstructured)
                available_actions = scenario.get_available_Red_actions()
        
                choose_action_result = adm.choose_action(
                    obs,
                    [deepcopy(a) for a in available_actions],
                    alignment_target if cfg.align_to_target else None,
                    scenario_id=scenario.id(),
                    **cfg.adm.get('inference_kwargs', {}))
                if isinstance(choose_action_result, tuple):
                    action_to_take, choice_info = choose_action_result
                    if 'choice_info' in choice_info:
                        # Handle pipeline_adm
                        choice_info = choice_info['choice_info']
                else:
                    action_to_take = choose_action_result
                    choice_info = {}

                print(action_to_take.name)

                if isinstance(action_to_take, dict):
                    log.info(json.dumps(action_to_take, indent=4), extra={"highlighter": JSON_HIGHLIGHTER})
                else:
                    log.info(json.dumps(action_to_take.to_dict(), indent=4), extra={"highlighter": JSON_HIGHLIGHTER})


                if action_to_take.name == 'Sleep':
                    blue_agent = blue_sleep
                elif action_to_take.name == 'Meander':
                    blue_agent = blue_meander
                elif action_to_take.name == 'B-line':   
                    blue_agent = blue_b_line
                # 

            if algorithm != 'LLM + RL':
                a = blue_agent.get_action(interface.wrapped_cyborg.vector_representation, None)
                action_to_take = possible_action[a]
                scenario.step(action_to_take)
            else:
                if s < 4:
                    available_actions = scenario.get_available_actions()
                    action_to_take = available_actions[0]
                    try:
                        if hasattr(action_to_take, "intent_action") and action_to_take.intent_action:
                            current_state = scenario.intend_action(action_to_take)
                        else:
                            current_state = scenario.take_action(action_to_take)
                    except Exception as e:
                        log.info(action_to_take)
                        raise e
                else:
                    a = blue_agent.get_action(interface.wrapped_cyborg.vector_representation, None)
                    action_to_take = possible_action[a]
                    scenario.step(action_to_take)
            s += 1
            sum_rewards += scenario.reward
            ep_rewards.append(scenario.reward)
            ep_confidentiality.append(scenario.cia_scores[0])
            ep_integrity.append(scenario.cia_scores[1])
            ep_availability.append(scenario.cia_scores[2])
            # ep_resiliences.append(scenario.resilience_score)
            scenario_complete = current_state.scenario_complete
        # print(current_state.unstructured)
        returns.append(sum_rewards)
        # log.info("*Sum of rewards in episode*: {}".format(sum_rewards))
        print("*Sum of rewards in episode*: {0} for {1}".format(sum_rewards, interface.red_agent.__name__))
        rewards.append(np.array(ep_rewards))
        confidentiality.append(np.array(ep_confidentiality))
        integrity.append(np.array(ep_integrity))
        availability.append(np.array(ep_availability))
        # resiliences.append(np.array(ep_resiliences))



        # log.info("[bold]*ACTION BEING TAKEN*[/bold]",
        #             extra={"markup": True})
        
        # log.info(action_to_take)
        
    # log.info('Red agents classification results:')
    # log.info(predictions)

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

    print('Average reward is: {0} with a standard deviation of {1}'.format(np.mean(returns), np.std(returns)))
if __name__ == "__main__":
    main()
