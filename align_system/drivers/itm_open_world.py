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
from align_system.utils.action_completion import DEFAULT_TAGS
from align_system.utils.version import get_version
from align_system.exceptions import SceneSkipException


log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


def as_dict(obj):
    return obj.to_dict() if hasattr(obj, "to_dict") else obj._asdict()


def compute_time_stats(times_s):
    n_times = len(times_s)
    total_time_s = sum(times_s)
    return {
        "n_actions_taken": n_times,
        "total_time_s": total_time_s,
        "avg_time_s": total_time_s / n_times if n_times else 0.,
        "max_time_s": max(times_s) if n_times else 0.,
        "raw_times_s": times_s
    }


def make_input_output_entry(scenario_id,
                            alignment_target_id,
                            current_state,
                            available_actions,
                            choice_info,
                            action_choice_idx,
                            action_to_take):
    # Capture inputs and outputs in a similar format to what's used by
    # our internal evaluation framework code
    return {'input': {'scenario_id': scenario_id,
                      'alignment_target_id': alignment_target_id,
                      'full_state': as_dict(current_state),
                      'state': current_state.unstructured,
                      'choices': [as_dict(a) for a in available_actions]},
            'label': [{} if a.kdma_association is None else a.kdma_association
                      for a in available_actions],
            'choice_info': choice_info,
            'output': {'choice': action_choice_idx,
                       'action': as_dict(action_to_take)}}


class ITMOpenWorldDriver:
    # Written to the run's meta.json sidecar as "driver" when set
    driver_name = None

    def __init__(self,
                apply_action_filtering=True,
                expand_actions=False,
                expand_tagging=False,
                sort_available_actions=False):
        self.apply_action_filtering = apply_action_filtering
        self.expand_actions = expand_actions
        self.expand_tagging = expand_tagging
        self.sort_available_actions = sort_available_actions

    def _expand_action_by_character(self, action, characters):
        expanded_actions = []

        for character in characters:
            new_action = deepcopy(action)
            new_action.character_id = character.id
            new_action.unstructured = re.sub(r"(a )?Patient", character.name, action.unstructured)

            expanded_actions.append(new_action)

        return expanded_actions

    def _expand_action_by_tag(self, action, possible_tags=DEFAULT_TAGS):
        assert action.action_type == ActionTypeEnum.TAG_CHARACTER

        expanded_actions = []

        for tag in possible_tags:
            new_action = deepcopy(action)
            if new_action.parameters is None:
                new_action.parameters = {}

            new_action.parameters['category'] = tag
            if re.match(r'^[aeiou]', tag, re.I):
                prefix = "an"
            else:
                prefix = "a"

            new_action.unstructured = re.sub(r"a triage tag", f"{prefix} {tag} triage tag", action.unstructured)

            expanded_actions.append(new_action)

        return expanded_actions

    def _get_expanded_and_filtered_actions(self,
                                           current_state,
                                           available_actions,
                                           treated_patients,
                                           evac_patients):
        """Expand (per-character / per-tag) and filter the available
        actions per this driver's settings.

        Returns (available_actions_expanded, available_actions_filtered).
        Note that END_SCENE is always excluded from the filtered list
        when `apply_action_filtering` is enabled; callers are expected
        to fall back to END_SCENE (from the expanded list) when the
        filtered list is empty.
        """
        if not self.expand_actions:
            available_actions_expanded = available_actions
        else:
            available_actions_expanded = []
            for idx, a in enumerate(available_actions):
                if a.action_type == ActionTypeEnum.TAG_CHARACTER:
                    tagging_by_character = self._expand_action_by_character(
                            action=a,
                            characters=current_state.characters
                    )
                    if self.expand_tagging:
                        # Expanding twice here, once for
                        # characters, and again for possible tags
                        for char_expanded_action in tagging_by_character:
                            available_actions_expanded.extend(self._expand_action_by_tag(
                                action=char_expanded_action))
                    else:
                        available_actions_expanded.extend(tagging_by_character)

                elif a.action_type == ActionTypeEnum.TREAT_PATIENT:
                    available_actions_expanded.extend(self._expand_action_by_character(
                        action=a,
                        characters=current_state.characters
                    ))


                elif a.action_type == ActionTypeEnum.MOVE_TO_EVAC:
                    available_actions_expanded.extend(self._expand_action_by_character(
                        action=a,
                        characters=current_state.characters
                    ))

                else:
                    available_actions_expanded.append(a)

            log.debug("[bold]*AVAILABLE ACTIONS EXPANDED*[/bold]",
                      extra={"markup": True})
            log.debug(json.dumps([as_dict(a) for a in available_actions_expanded], indent=4),
                      extra={"highlighter": JSON_HIGHLIGHTER})

        if not self.apply_action_filtering:
            available_actions_filtered = available_actions_expanded
        else:
            available_actions_filtered = []
            for a in available_actions_expanded:
                if a.action_type == ActionTypeEnum.END_SCENE:
                    # We want to restrict end scene until all characters have been treated
                    continue

                elif a.action_type == ActionTypeEnum.TAG_CHARACTER:
                    untagged_characters = {
                        c.id for c in current_state.characters
                        if c.tag is None and not c.unseen
                    }
                    if len(untagged_characters) == 0:  # No more patients to tag
                        continue
                    if a.character_id is not None and a.character_id not in untagged_characters:
                        continue

                # HACK: Current TA3 server doesn't track what patients have been
                # treated or evac'd (via c.unseen, or any other means); need to
                # track it manually
                elif a.action_type == ActionTypeEnum.TREAT_PATIENT:
                    treatable_patients = {
                        c.id for c in current_state.characters
                        if c.id not in treated_patients
                    }
                    if len(treatable_patients) == 0:  # No more patients to treat
                        continue
                    if a.character_id is not None and a.character_id not in treatable_patients:
                        continue

                elif a.action_type == ActionTypeEnum.MOVE_TO_EVAC:
                    evacable_patients = {
                        c.id for c in current_state.characters
                        if c.id not in evac_patients
                    }
                    if len(evacable_patients) == 0:  # No more patients to evac
                        continue
                    if a.character_id is not None and a.character_id not in evacable_patients:
                        continue

                available_actions_filtered.append(a)

            log.debug("[bold]*AVAILABLE ACTIONS FILTERED*[/bold]",
                      extra={"markup": True})
            log.debug(json.dumps([as_dict(a) for a in available_actions_filtered], indent=4),
                      extra={"highlighter": JSON_HIGHLIGHTER})

        return available_actions_expanded, available_actions_filtered

    @staticmethod
    def _end_scene_fallback_action(available_actions_expanded):
        """Return the END_SCENE action to fall back to when the
        filtered action list is empty (END_SCENE is excluded from the
        filtered list whenever `apply_action_filtering` is
        enabled)."""
        for a in available_actions_expanded:
            if a.action_type == ActionTypeEnum.END_SCENE:
                log.info("** All patients have been tagged and treated, ending scene")
                return a

        raise RuntimeError("No available actions from filtered list!")

    def _initialize_run(self, cfg):
        """One-time setup before any scenario is started."""
        adm = cfg.adm.instance

        # HACK: need to invoke 'load_model' for ADMs that require it,
        # maybe it makes more sense to load_model in the init method for
        # those ADMs
        if hasattr(adm, 'load_model'):
            adm.load_model()

    def _run_scenario(self, cfg, scenario, alignment_target,
                      sort_available_actions, record_input_output):
        """Drive a single scenario: repeatedly delegate the choice among
        the available actions to the configured ADM until the scenario
        is complete.

        `record_input_output(entry)` is called with a
        `make_input_output_entry` record after each action taken.

        Returns (per-action decision times, final state,
        scenario_complete).
        """
        adm = cfg.adm.instance

        # Reset any decision or chat history for a new scenario
        if hasattr(adm, 'reset_history'):
            log.info("[bold]*Resetting choice history*[/bold]")
            adm.reset_history()

        current_state = scenario.get_state()
        scenario_complete = current_state.scenario_complete

        sce_times_s = []

        last_scene_id = None

        treated_patients = set()
        evac_patients = set()

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
            log.debug(json.dumps([as_dict(a) for a in available_actions], indent=4),
                      extra={"highlighter": JSON_HIGHLIGHTER})

            available_actions_expanded, available_actions_filtered =\
                self._get_expanded_and_filtered_actions(
                    current_state,
                    available_actions,
                    treated_patients,
                    evac_patients)

            if len(available_actions_filtered) == 0:
                action_to_take = self._end_scene_fallback_action(
                    available_actions_expanded)
                action_to_take.justification = "All patients have been tagged and treated"
                choice_info = {}
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
                log.info(json.dumps(as_dict(action_to_take), indent=4),
                         extra={"highlighter": JSON_HIGHLIGHTER})

            action_choice_idx = None
            for i, a in enumerate(available_actions):
                if a.action_id == action_to_take.action_id:
                    action_choice_idx = i
                    break

            # Ensure that 'actions' stored in 'choice_info' are serializable
            for info in choice_info.values():
                if isinstance(info, dict) and 'action' in info:
                    info['action'] = info['action'].to_dict()

            record_input_output(make_input_output_entry(
                scenario_id=scenario.id(),
                alignment_target_id=(alignment_target.id
                                     if cfg.align_to_target else None),
                current_state=current_state,
                available_actions=available_actions,
                choice_info=choice_info,
                action_choice_idx=action_choice_idx,
                action_to_take=action_to_take))

            try:
                if hasattr(action_to_take, "intent_action") and action_to_take.intent_action:
                    current_state = scenario.intend_action(action_to_take)
                else:
                    current_state = scenario.take_action(action_to_take)
            except Exception as e:
                if hasattr(e, 'json'):
                    log.info(e.json(indent=2))
                else:
                    log.info(str(e))
                raise e

            # If we treated a patient, record that treatment so we can ensure we treat everyone
            if action_to_take.action_type == ActionTypeEnum.TREAT_PATIENT:
                treated_patients.add(action_to_take.character_id)
            # If we evaced a patient, record that so we don't try to evac them again
            if action_to_take.action_type == ActionTypeEnum.MOVE_TO_EVAC:
                evac_patients.add(action_to_take.character_id)

            scenario_complete = current_state.scenario_complete

        return sce_times_s, current_state, scenario_complete

    def drive(self, cfg):
        interface = cfg.interface

        self._initialize_run(cfg)

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

        inputs_outputs = []

        def record_input_output(entry):
            # Save input_output after each action (gets overwritten
            # each time) so that we don't lose everything if the run
            # crashes or is interrupted.  Could treat this as we do
            # the logfile and open the file handle once and close
            # `atexit` and write each line as it's generated (and make
            # it a .jsonl file; would need to remove the indent=2)
            inputs_outputs.append(entry)
            if save_input_output_to_path is not None:
                with open(save_input_output_to_path, 'w') as f:
                    json.dump(inputs_outputs, f, indent=2)

        # Write version sidecar once at the start of the run
        meta = {"version": get_version()}
        if self.driver_name is not None:
            meta["driver"] = self.driver_name
        username = getattr(interface, 'username', None)
        if username is not None:
            meta["username"] = username
        with open(os.path.join(output_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        session_alignment_scores = []

        # Capture time it takes to choose each action
        action_times = { "scenarios": [] }

        # Loop through available scenarios
        while scenario := interface.start_scenario():
            if scenario.id() == '':
                log.info("Next scenario ID is blank, assuming we're done, exiting")
                break
            log.info(f'[bold]*Scenario ID*[/bold]: {scenario.id()}')

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

            sce_times_s, final_state, scenario_complete = self._run_scenario(
                cfg, scenario, alignment_target,
                sort_available_actions, record_input_output)

            if scenario_complete:
                log.info("*Final state unstructured*: {}".format(
                    final_state.unstructured))

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
                        print(final_state.unstructured, file=f)

            if save_timing_to_path is not None:
                action_times["scenarios"].append(compute_time_stats(sce_times_s))

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

            action_times.update(compute_time_stats(all_times))

            with open(save_timing_to_path, 'w') as f:
                json.dump(action_times, f, indent=2)

        if len(session_alignment_scores) > 0:
            if save_alignment_score_to_path is not None:
                with open(save_alignment_score_to_path, 'w') as f:
                    json.dump([(s if isinstance(s, dict) else s.to_dict())
                               for s in session_alignment_scores], f, indent=2)
