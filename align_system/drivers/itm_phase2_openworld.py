import os
import json
import random
from copy import deepcopy

from rich.highlighter import JSONHighlighter
import hydra
from omegaconf import DictConfig, OmegaConf
from timeit import default_timer as timer

from align_system.utils import logging
from align_system.utils.version import get_version
from align_system.exceptions import SceneSkipException

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class ITMPhase2OpenWorldDriver:
    """Driver for open-world experiments where action filtering is disabled.

    All available actions are passed to the ADM without any server-side
    filtering.  The ADM's conversation history (e.g. BasicOpenWorldDialogADMComponent)
    is expected to prevent redundant re-treatment of characters on its own.
    """

    def drive(self, cfg):
        interface = cfg.interface
        adm = cfg.adm.instance

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

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

        if hasattr(adm, 'load_model'):
            adm.load_model()

        inputs_outputs = []

        meta = {"version": get_version()}
        username = getattr(interface, 'username', None)
        if username is not None:
            meta["username"] = username
        with open(os.path.join(output_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        session_alignment_scores = []

        action_times = {"scenarios": []}
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

        while scenario := interface.start_scenario():
            if scenario.id() == '':
                log.info("Next scenario ID is blank, assuming we're done, exiting")
                break
            log.info(f'[bold]*Scenario ID*[/bold]: {scenario.id()}')

            if hasattr(adm, 'reset_history'):
                log.info("[bold]*Resetting choice history*[/bold]")
                adm.reset_history()

            if 'alignment_target' in cfg:
                alignment_target = cfg.alignment_target
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

            sce_times_s = []

            last_scene_id = None

            while not scenario_complete:
                current_scene_id = current_state.meta_info.scene_id
                if last_scene_id != current_scene_id:
                    log.info(f"[bold]*CHANGED SCENE TO*: {current_scene_id}[/bold]",
                             extra={"markup": True})
                    last_scene_id = current_scene_id

                available_actions = scenario.get_available_actions()

                log.debug("[bold]*AVAILABLE ACTIONS*[/bold]",
                          extra={"markup": True})
                log.debug(json.dumps([a.to_dict() if hasattr(a, "to_dict") else a._asdict() for a in available_actions], indent=4),
                          extra={"highlighter": JSON_HIGHLIGHTER})

                # All actions are passed through — the ADM's dialog history
                # is responsible for avoiding re-treatment.
                available_actions_filtered = available_actions

                if len(available_actions_filtered) == 0:
                    raise RuntimeError("No available actions!")
                elif len(available_actions_filtered) == 1:
                    log.info("** Choosing only available action")
                    action_to_take = available_actions_filtered[0]
                    action_to_take.justification = "Only available action"
                    choice_info = {}
                else:
                    start_choose_action = timer()

                    try:
                        choose_action_result = adm.choose_action(
                            current_state,
                            [deepcopy(a) for a in available_actions_filtered],
                            alignment_target if cfg.align_to_target else None,
                            scenario_id=scenario.id(),
                            **cfg.adm.get('inference_kwargs', {}))

                        if isinstance(choose_action_result, tuple):
                            action_to_take, choice_info = choose_action_result
                            if 'choice_info' in choice_info:
                                choice_info = choice_info['choice_info']
                        else:
                            action_to_take = choose_action_result
                            choice_info = {}

                    except SceneSkipException as e:
                        log.error(f"Scene skipped due to component failure: {e}")
                        log.info(f"Component {e.component_name} failed - choosing random action to advance scene")

                        action_to_take = random.choice(available_actions_filtered)
                        action_to_take.justification = f"Random action chosen due to component failure: {e.component_name}"
                        choice_info = {}

                        log.warning(f"Taking random action to advance: {action_to_take.action_type if hasattr(action_to_take, 'action_type') else 'unknown'}")

                    end_choose_action = timer()
                    sce_times_s.append(end_choose_action - start_choose_action)
                    log.debug(f"choose_action took {end_choose_action - start_choose_action} seconds")

                log.info("[bold]*ACTION BEING TAKEN*[/bold]",
                         extra={"markup": True})
                if isinstance(action_to_take, dict):
                    log.info(json.dumps(action_to_take, indent=4),
                             extra={"highlighter": JSON_HIGHLIGHTER})
                else:
                    log.info(json.dumps(action_to_take.to_dict() if hasattr(action_to_take, "to_dict") else action_to_take._asdict(), indent=4),
                             extra={"highlighter": JSON_HIGHLIGHTER})

                action_choice_idx = None
                for i, a in enumerate(available_actions):
                    if a.action_id == action_to_take.action_id:
                        action_choice_idx = i
                        break

                for info in choice_info.values():
                    if 'action' in info:
                        info['action'] = info['action'].to_dict()

                inputs_outputs.append({'input': {'scenario_id': scenario.id(),
                                                 'alignment_target_id': alignment_target.id if cfg.align_to_target else None,
                                                 'full_state': current_state.to_dict() if hasattr(current_state, "to_dict") else current_state._asdict(),
                                                 'state': current_state.unstructured,
                                                 'choices': [a.to_dict() if hasattr(a, "to_dict") else a._asdict() for a in available_actions]},
                                       'label': [{} if a.kdma_association is None else a.kdma_association for a in available_actions],
                                       'choice_info': choice_info,
                                       'output': {'choice': action_choice_idx,
                                                  'action': action_to_take.to_dict() if hasattr(action_to_take, "to_dict") else action_to_take._asdict()}})

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

            if save_timing_to_path is not None:
                action_times["scenarios"].append(_compute_time_stats(sce_times_s))

            if alignment_target is not None:
                try:
                    session_alignment = interface.get_session_alignment(alignment_target)
                except Exception:
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
