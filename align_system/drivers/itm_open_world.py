import os
import json
import random
import re
from copy import deepcopy

from rich.highlighter import JSONHighlighter
import hydra
from omegaconf import DictConfig, OmegaConf
from swagger_client.models import ActionTypeEnum
from timeit import default_timer as timer

from align_system.utils import logging
from align_system.utils.version import get_version
from align_system.exceptions import SceneSkipException

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class ITMOpenWorldDriver:
    def __init__(self,
                apply_action_filtering=True,
                sort_available_actions=False):
        self.apply_action_filtering = apply_action_filtering
        self.sort_available_actions = sort_available_actions

    def _expand_action_by_character(self, action, characters, restricted_character_ids):
        expanded_actions = []

        for character in characters:
            if character.id in restricted_character_ids:
                continue

            new_action = deepcopy(action)
            new_action.character_id = character.id
            new_action.unstructured = re.sub(r"(a )?Patient", character.name, action.unstructured)

            expanded_actions.append(new_action)

        return expanded_actions


    def drive(self, cfg):
        interface = cfg.interface
        adm = cfg.adm.instance

        # Using the hydra generated output directory for the run
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

        if cfg.get('force_determinism', False) or self.sort_available_actions:
            log.info("Setting `sort_available_actions` to True")
            sort_available_actions = True
        else:
            sort_available_actions = False

        # HACK: need to invoke 'load_model' for ADMs that require it,
        # maybe it makes more sense to load_model in the init method for
        # those ADMs
        if hasattr(adm, 'load_model'):
            adm.load_model()

        # Capture inputs and outputs in a similar format to what's used by
        # our internal evaluation framework code
        inputs_outputs = []

        # Write version sidecar once at the start of the run
        meta = {"version": get_version()}
        username = getattr(interface, 'username', None)
        if username is not None:
            meta["username"] = username
        with open(os.path.join(output_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

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

            sce_times_s = []

            last_scene_id = None

            treated_patients = []
            evac_patients = []

            while not scenario_complete:
                current_scene_id = current_state.meta_info.scene_id
                if last_scene_id != current_scene_id:
                    log.info(f"[bold]*CHANGED SCENE TO*: {current_scene_id}[/bold]",
                             extra={"markup": True})
                    last_scene_id = current_scene_id

                available_actions = scenario.get_available_actions()

                if sort_available_actions:
                    # Impose a fixed ordering of available actions to help
                    # with determinism
                    available_actions = sorted(available_actions, key=lambda a: a.unstructured)

                log.debug("[bold]*AVAILABLE ACTIONS*[/bold]",
                          extra={"markup": True})
                log.debug(json.dumps([a.to_dict() if hasattr(a, "to_dict") else a._asdict() for a in available_actions], indent=4),
                          extra={"highlighter": JSON_HIGHLIGHTER})

                if not self.apply_action_filtering:
                    available_actions_filtered = available_actions
                else:
                    available_actions_filtered = []
                    end_scene_idx = None
                    for idx, a in enumerate(available_actions):
                        if a.action_type == ActionTypeEnum.END_SCENE:
                            # We want to restrict end scene until all characters have been treated
                            end_scene_idx = idx
                            continue

                        if a.action_type == ActionTypeEnum.TAG_CHARACTER:
                            # Don't let ADM choose to tag a character unless there are
                            # still untagged characters
                            untagged_characters = [c for c in current_state.characters
                                                if c.tag is None and not c.unseen]

                            available_actions_filtered.extend(self._expand_action_by_character(
                                action=a,
                                characters=untagged_characters,
                                restricted_character_ids=[],
                            ))

                        if a.action_type == ActionTypeEnum.TREAT_PATIENT:
                            treatable_patients = [c for c in current_state.characters if not c.unseen]

                            available_actions_filtered.extend(self._expand_action_by_character(
                                action=a,
                                characters=treatable_patients,
                                restricted_character_ids=treated_patients,
                            ))


                        if a.action_type == ActionTypeEnum.MOVE_TO_EVAC:
                            evacable_patients = [c for c in current_state.characters if not c.unseen]

                            available_actions_filtered.extend(self._expand_action_by_character(
                                action=a,
                                characters=evacable_patients,
                                restricted_character_ids=evac_patients,
                            ))

                if len(available_actions_filtered) == 0:
                    if end_scene_idx is not None:  # All patients have been tagged and treated
                        log.info("** All patients have been tagged and treated, ending scene")
                        action_to_take = available_actions[end_scene_idx]
                        action_to_take.justification = "All patients have been tagged and treated"
                    else:
                        raise RuntimeError("No available actions from filtered list!")
                elif len(available_actions_filtered) == 1:
                    log.info("** Choosing only available (filtered) action")
                    action_to_take = available_actions_filtered[0]
                    action_to_take.justification = "Only available (filtered) action"
                else:
                    start_choose_action = timer()

                    try:
                        # Passing in a copy of available actions to
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

                    except SceneSkipException as e:
                        log.error(f"Scene skipped due to component failure: {e}")
                        log.info(f"Component {e.component_name} failed - choosing random action to advance scene")

                        # Choose a random action from available_actions_filtered to advance the scenario
                        action_to_take = random.choice(available_actions_filtered)
                        action_to_take.justification = f"Random action chosen due to component failure: {e.component_name}"
                        choice_info = {}

                        log.warning(f"Taking random action to advance: {action_to_take.action_type if hasattr(action_to_take, 'action_type') else 'unknown'}")

                    # Common code for both success and exception paths
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

                # Ensure that 'actions' stored in 'choice_info' are serializable
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
                # Save input_output after each action (gets overwritten
                # each time) so that we don't lose everything if the run
                # crashes or is interrupted.  Could treat this as we do
                # the logfile and open the file handle once and close
                # `atexit` and write each line as it's generated (and make
                # it a .jsonl file; would need to remove the indent=2)
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

                # If we treated a patient, record that treatment so we can ensure we treat everyone
                if action_to_take.action_type == ActionTypeEnum.TREAT_PATIENT:
                    treated_patients.append(action_to_take.character_id)
                # If we evaced a patient, record that so we don't try to evac them again
                if action_to_take.action_type == ActionTypeEnum.MOVE_TO_EVAC:
                    evac_patients.append(action_to_take.character_id)

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
