"""Open-world harness supplying environment tools to a configured agentic ADM.

The harness owns action validation, environment state, scenario traversal and
run artifacts. The ADM owns model calls, treatment selection and review policy.
"""

import json
import os
from timeit import default_timer as timer

import hydra
from langchain_core.tools import tool
from omegaconf import DictConfig, ListConfig, OmegaConf
from swagger_client.models import ActionTypeEnum

from align_system.drivers.utils import open_world_actions as actions
from align_system.drivers.utils.open_world_actions import (
    ActionResolutionSnapshot, TOOL_NAMES_BY_ACTION_TYPE,
    as_dict as _as_dict, enum_value as _enum_value,
    filter_available_actions as _filter_available_actions,
    untagged_characters as _untagged_characters,
    format_available_actions as _format_available_actions,
    resolve_scene_end_policy,
)
from align_system.drivers.utils.open_world_patient_history import PatientObservationHistory
from align_system.exceptions import ActionRejectedException
from align_system.utils import logging
from align_system.utils.agent_logging import agent_log_event, agent_log_private, preview
from align_system.utils.version import get_version

log = logging.getLogger(__name__)

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


def _action_log_view(action):
    """Only action fields visible to the agent, without evaluation labels."""
    return {key: value for key, value in _as_dict(action).items() if key in {
        'action_id', 'action_type', 'character_id', 'parameters',
        'unstructured', 'justification', 'intent_action',
    }}


def _describe_visible_character(character):
    return {key: value for key, value in _as_dict(character).items()
            if value is not None and key not in {'has_blanket', 'kdma_association'}}


def _observation_view(current_state):
    """The same visible state supplies tools, history, and review evidence."""
    observation = {
        'scene_id': current_state.meta_info.scene_id,
        'situation': current_state.unstructured,
        'casualties': [_describe_visible_character(c) for c in current_state.characters
                       if not getattr(c, 'unseen', False)],
    }
    supplies = getattr(current_state, 'supplies', None)
    observation['supplies'] = (None if supplies is None else [
        {'type': _enum_value(s.type), 'quantity': s.quantity} for s in supplies])
    if getattr(current_state, 'environment', None) is not None:
        environment = current_state.environment
        observation['environment'] = (
            environment.to_dict() if hasattr(environment, 'to_dict') else environment)
    # Match the JSON-safe values exposed by observe_environment, without I/O.
    return json.loads(json.dumps(observation, default=str))


class ScenarioSession:
    """Mutable per-scenario state behind the agent's tools: the current
    environment state, manual treated/evac'd/tagged patient tracking,
    the most recently listed actions, and input/output bookkeeping for
    each executed action."""

    action_types = TOOL_NAMES_BY_ACTION_TYPE
    action_tool_names = {name: kind for kind, name in TOOL_NAMES_BY_ACTION_TYPE.items()}
    injury_locations = actions.INJURY_LOCATIONS
    format_available_actions = staticmethod(actions.format_available_actions)
    proposal_tool_view = staticmethod(actions.proposal_tool_view)
    review_action_view = staticmethod(actions.review_action_view)
    same_action = staticmethod(actions.same_action)
    resolve_typed_action = staticmethod(actions.resolve_typed_action)
    enum_value = staticmethod(actions.enum_value)
    in_stock_supplies = staticmethod(actions.in_stock_supplies)

    def snapshot(self):
        """Copy current offers and resolver state without observing again."""
        return ActionResolutionSnapshot.from_session(self)

    def observe(self, state=None):
        return _observation_view(self.current_state if state is None else state)

    def resolve_action(self, tool_name, arguments, *, snapshot=None):
        """Resolve native arguments without submitting the proposed action."""
        if snapshot is None:
            self.refresh_actions()
            snapshot = self.snapshot()
        return actions.resolve_typed_action(
            snapshot, self.action_tool_names[tool_name], **arguments)

    def parameter_validation(self, action, checked=True, *, snapshot=None):
        return actions.parameter_validation(
            self if snapshot is None else snapshot, action, checked=checked)

    def __init__(self, scenario, alignment_target, apply_action_filtering,
                 sort_available_actions, record_input_output,
                 scene_end_policy='auto', max_actions_per_scenario=100,
                 align_to_target=None):
        self.scenario = scenario
        self.alignment_target = alignment_target
        self.align_to_target = (alignment_target is not None if align_to_target is None
                                else align_to_target)
        self.scene_end_policy = resolve_scene_end_policy(scene_end_policy, self.align_to_target)
        self.max_actions_per_scenario = max_actions_per_scenario
        self.native_tools = {}
        self.native_action_schemas = {}
        self.apply_action_filtering = apply_action_filtering
        self.sort_available_actions = sort_available_actions
        self.record_input_output = record_input_output

        self.current_state = scenario.get_state()
        self.scenario_complete = self.current_state.scenario_complete
        self.treated_patients = set()
        self.evac_patients = set()
        self.tagged_patients = set()
        self.available_actions = []
        self.actions_filtered = []
        self.actions_observed_at = None
        self.scene_done = False
        self.n_actions = 0
        self.patient_history = PatientObservationHistory()
        self.patient_history.update(_observation_view(self.current_state), self.n_actions)
        self.times_s = []
        self.decision_start = timer()

    def log_context(self):
        """Read correlation identifiers without observing or changing state."""
        return {
            'scenario_id': self.scenario.id(),
            'target_id': getattr(self.alignment_target, 'id', None),
            'scene_id': self.current_state.meta_info.scene_id,
            'action_index': self.n_actions,
        }

    def all_patients_handled(self):
        """True once every (seen) character is tagged and every
        character has been treated."""
        untagged = _untagged_characters(
            self.current_state, self.tagged_patients)
        untreated = {c.id for c in self.current_state.characters
                     if c.id not in self.treated_patients}
        return len(untagged) == 0 and len(untreated) == 0

    def refresh_actions(self):
        """Re-fetch and filter the environment's available actions,
        updating `available_actions` / `actions_filtered` /
        `scene_done`; returns the filtered list.

        Legacy scenes close after all care or when filtering leaves no other
        actions. In agent_decides mode, offered END_SCENE remains eligible and
        scene_done never requests automatic closure."""
        available_actions = self.scenario.get_available_actions()

        if self.sort_available_actions:
            # Impose a fixed ordering of available actions to help
            # with determinism
            available_actions = sorted(
                available_actions, key=lambda a: a.unstructured)

        agent_log_event(
            log, 'ENVIRONMENT', 'AVAILABLE ACTIONS',
            f'offered={len(available_actions)}', level=logging.DEBUG,
            payload=lambda: [_action_log_view(a) for a in available_actions],
            **self.log_context())

        if self.apply_action_filtering:
            end_scene_actions = [a for a in available_actions
                                 if a.action_type == ActionTypeEnum.END_SCENE]
            filtered = _filter_available_actions(
                self.current_state,
                [a for a in available_actions
                 if a.action_type != ActionTypeEnum.END_SCENE],
                self.treated_patients, self.evac_patients,
                self.tagged_patients)

            self.scene_done = (self.scene_end_policy == 'legacy'
                               and (self.all_patients_handled() or not filtered))
            if self.scene_done or self.scene_end_policy == 'agent_decides':
                filtered.extend(end_scene_actions)

            agent_log_event(
                log, 'ENVIRONMENT', 'ELIGIBLE ACTIONS',
                f'eligible={len(filtered)}', level=logging.DEBUG,
                payload=lambda: [_action_log_view(a) for a in filtered],
                **self.log_context())
        else:
            filtered = list(available_actions)
            self.scene_done = (self.scene_end_policy == 'legacy'
                               and self.all_patients_handled())

        self.available_actions = available_actions
        self.actions_filtered = filtered
        self.actions_observed_at = self.n_actions

        return filtered

    def execute(self, action_to_take, justification=None, choice_info=None,
                decision_time_s=None):
        """Submit an action to the environment, record it, and update
        the session state; raises ActionRejectedException when the
        environment refuses the action (recoverable by choosing a
        different action).

        `decision_time_s`, when given, is appended to the per-scenario
        timing stats (`times_s`)."""
        if justification and getattr(
                action_to_take, 'justification', None) is None:
            action_to_take.justification = justification

        action_context = self.log_context()
        action_summary = (
            f'action={_enum_value(action_to_take.action_type)} '
            f'patient={getattr(action_to_take, "character_id", None)} '
            f'intent={bool(getattr(action_to_take, "intent_action", False))}')
        agent_log_event(
            log, 'ENVIRONMENT', 'SUBMIT', action_summary,
            payload=lambda: _action_log_view(action_to_take), **action_context)

        try:
            if getattr(action_to_take, "intent_action", False):
                current_state = self.scenario.intend_action(action_to_take)
            else:
                current_state = self.scenario.take_action(action_to_take)
        except Exception as e:
            # The environment may return private transport details. Keep the
            # existing model feedback while excluding these details from logs.
            agent_log_private(str(getattr(e, 'body', None) or e))
            agent_log_event(
                log, 'ENVIRONMENT', 'REJECTED',
                f'{action_summary} error_type={type(e).__name__} '
                f'status={getattr(e, "status", None)}',
                level=logging.WARNING, **action_context)

            if getattr(e, 'status', None) in (400, 500):
                # The environment refused the action -- 400 for e.g. a
                # too-distant character, 500 when the (live) server
                # chokes on the action's parameters (e.g. TREAT_PATIENT
                # without a treatment supply); recoverable by choosing
                # differently
                raise ActionRejectedException(
                    str(getattr(e, 'body', e))) from e
            raise e

        # Only successfully executed actions are recorded
        if decision_time_s is not None:
            self.times_s.append(decision_time_s)
        self._record_action(action_to_take, choice_info or {})
        self.patient_history.record_tag(
            action_to_take, self.current_state.meta_info.scene_id, self.n_actions)

        if action_to_take.action_type == ActionTypeEnum.TREAT_PATIENT:
            self.treated_patients.add(action_to_take.character_id)
        if action_to_take.action_type == ActionTypeEnum.MOVE_TO_EVAC:
            self.evac_patients.add(action_to_take.character_id)
        if action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            self.tagged_patients.add(action_to_take.character_id)

        if (current_state.meta_info.scene_id
                != self.current_state.meta_info.scene_id):
            # A new scene presents fresh injuries (and drops tags), so
            # the manually tracked per-scene bookkeeping is stale;
            # without this reset the filter would drop every action
            # once each patient had been treated / tagged once
            agent_log_event(
                log, 'ENVIRONMENT', 'SCENE CHANGED',
                f'{self.current_state.meta_info.scene_id} -> '
                f'{current_state.meta_info.scene_id}; resetting patient tracking',
                **action_context)
            self.treated_patients.clear()
            self.evac_patients.clear()
            self.tagged_patients.clear()

        self.current_state = current_state
        self.scenario_complete = current_state.scenario_complete
        self.n_actions += 1
        self.patient_history.update(_observation_view(current_state), self.n_actions)
        # Listed actions are stale after the environment changes
        self.actions_filtered = []
        self.actions_observed_at = None
        self.decision_start = timer()

        agent_log_event(
            log, 'ENVIRONMENT', 'EXECUTED',
            f'{action_summary} scenario_complete={self.scenario_complete}',
            **action_context)

        return current_state

    def _record_action(self, action_to_take, choice_info):
        # Called before the session state is updated, so the recorded
        # state/choices are the ones the decision was made against
        action_choice_idx = None
        for i, a in enumerate(self.available_actions):
            if a.action_id == action_to_take.action_id:
                action_choice_idx = i
                break

        # Ensure that 'actions' stored in 'choice_info' are serializable
        for info in choice_info.values():
            if isinstance(info, dict) and 'action' in info:
                info['action'] = _as_dict(info['action'])

        # Same format as the ADM driver's input_output.json entries
        # (and our internal evaluation framework code)
        self.record_input_output({
            'input': {'scenario_id': self.scenario.id(),
                      'alignment_target_id': (
                          self.alignment_target.id
                          if self.alignment_target is not None else None),
                      'full_state': _as_dict(self.current_state),
                      'state': self.current_state.unstructured,
                      'choices': [_as_dict(a)
                                  for a in self.available_actions]},
            'label': [{} if a.kdma_association is None
                      else a.kdma_association
                      for a in self.available_actions],
            'choice_info': choice_info,
            'output': {'choice': action_choice_idx,
                       'action': _as_dict(action_to_take)}})


class ITMOpenWorldAgenticDriver:
    """Traverse scenarios and provide native tools to ``cfg.adm.instance``."""

    def __init__(self, max_actions_per_scenario=100, apply_action_filtering=True,
                 sort_available_actions=False, extra_tools=None, scene_end_policy='auto'):
        self.max_actions_per_scenario = max_actions_per_scenario
        self.apply_action_filtering = apply_action_filtering
        self.sort_available_actions = sort_available_actions
        self.extra_tools = dict(extra_tools) if extra_tools else {}
        resolve_scene_end_policy(scene_end_policy, False)
        self.scene_end_policy = scene_end_policy

    def _observation_text(self, current_state):
        return json.dumps(_observation_view(current_state), indent=2)

    def _perform_typed_action(self, session, action_type, justification, **arguments):
        action, error = session.resolve_action(
            session.action_types[action_type], {'justification': justification, **arguments})
        if error is not None:
            return error
        try:
            current_state = session.execute(
                action, justification, decision_time_s=timer() - session.decision_start)
        except ActionRejectedException as error:
            return f"The environment rejected this action: {error}. Choose a different action."
        if current_state.scenario_complete:
            return "Action executed.  SCENARIO COMPLETE -- you are done."
        return "Action executed.  Updated environment:\n" + self._observation_text(current_state)

    def _build_tools(self, session):
        """Build the LangChain tools through which the agent interacts
        with the per-scenario `session`: two observation tools plus one
        tool per action type in the pipeline (see
        TOOL_NAMES_BY_ACTION_TYPE)."""
        driver = self

        @tool
        def observe_environment() -> str:
            """Observe the current scene: the situation description and
            the casualties (with their injuries, vitals, and triage
            tags)."""
            return self._observation_text(session.current_state)

        @tool
        def list_available_actions() -> str:
            """List the actions the environment currently offers, named
            by the tool that carries each one out."""
            listing = _format_available_actions(session.refresh_actions())

            return listing

        @tool
        def check_vitals(character_name: str, justification: str) -> str:
            """Check the vitals of a casualty.

            Args:
                character_name: the casualty whose vitals to check
                justification: brief clinical reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.CHECK_VITALS, justification,
                character_name=character_name)

        @tool
        def treat_patient(character_name: str, justification: str,
                          treatment_supply: str = "",
                          injury_location: str = "") -> str:
            """Treat a casualty's injuries with a supply from your
            inventory.

            Args:
                character_name: the casualty to treat
                justification: brief clinical reasoning for this action
                treatment_supply: the supply to treat with (one of the
                    supplies listed in your observation, e.g.
                    'Tourniquet', 'Pressure bandage', 'Hemostatic
                    gauze'); when inventory is unknown, still name a
                    concrete supply, never 'null', 'none', or 'unknown'
                injury_location: where on the body the injury being
                    treated is (e.g. 'left calf', 'right thigh',
                    'center chest'; 'unspecified' if unclear)
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.TREAT_PATIENT, justification,
                character_name=character_name,
                treatment_supply=treatment_supply,
                injury_location=injury_location)

        @tool
        def tag_character(character_name: str, triage_tag: str,
                          justification: str) -> str:
            """Apply a triage tag to a casualty.

            Args:
                character_name: the casualty to tag
                triage_tag: the triage category to apply (MINIMAL,
                    DELAYED, IMMEDIATE, or EXPECTANT)
                justification: brief clinical reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.TAG_CHARACTER, justification,
                character_name=character_name, triage_tag=triage_tag)

        @tool
        def move_to(character_name: str, justification: str) -> str:
            """Move to a casualty (often required before they can be
            assessed or treated).

            Args:
                character_name: the casualty to move to
                justification: brief clinical reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.MOVE_TO, justification,
                character_name=character_name)

        @tool
        def move_to_evac(character_name: str, justification: str) -> str:
            """Move a casualty to evacuation.

            Args:
                character_name: the casualty to evacuate
                justification: brief clinical reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.MOVE_TO_EVAC, justification,
                character_name=character_name)

        @tool
        def search(justification: str) -> str:
            """Search the area for additional casualties.

            Args:
                justification: brief reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.SEARCH, justification)

        @tool
        def send_message(justification: str) -> str:
            """Deliver the message/communication the environment
            currently offers.

            Args:
                justification: brief reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.MESSAGE, justification)

        @tool
        def end_scene(justification: str) -> str:
            """End the current scene when eligible under the supplied scene
            ending policy. Explain why ending is appropriate.

            Args:
                justification: brief reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.END_SCENE, justification)

        tools = [observe_environment, list_available_actions,
                 check_vitals, treat_patient, tag_character, move_to,
                 move_to_evac, search, send_message, end_scene]
        session.native_tools = {item.name: item for item in tools}
        session.native_action_schemas = {
            t.name: t.args_schema for t in tools if t.name in session.action_tool_names}
        for name, provider in self.extra_tools.items():
            extra = list(provider.build_tools(session))
            log.info(f"Toolkit '{name}' added tools: "
                     f"{', '.join(t.name for t in extra)}")
            tools.extend(extra)
        return tools

    @staticmethod
    def _get_alignment_target(cfg, scenario):
        # An explicit target takes precedence, including for scoring
        # when alignment is disabled. Only aligned sessions expose it to agents
        # and attached toolkits.
        if 'alignment_target' in cfg:
            alignment_target = cfg.alignment_target
            # Alignment targets specified in hydra configs require
            # some nested conversion to dict (from OmegaConf objects)
            # otherwise this can cause some downstream issues with
            # serialization
            values = getattr(alignment_target, 'kdma_values', None)
            if isinstance(values, (list, tuple, ListConfig)):
                alignment_target.kdma_values = [OmegaConf.to_container(c)
                                                if isinstance(c, DictConfig) else c
                                                for c in values]
        elif cfg.align_to_target:
            alignment_target = scenario.get_alignment_target()
        else:
            alignment_target = None

        return alignment_target

    def drive(self, cfg):
        interface = cfg.interface

        adm = cfg.adm.instance
        if hasattr(adm, 'load_model'):
            adm.load_model()

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
        if save_input_output_to_path is not None:
            # A run with no successful actions still has a valid output artifact.
            with open(save_input_output_to_path, 'w') as f:
                json.dump(inputs_outputs, f, indent=2)

        def record_input_output(entry):
            # Save input_output after each action (gets overwritten
            # each time) so that we don't lose everything if the run
            # crashes or is interrupted
            inputs_outputs.append(entry)
            if save_input_output_to_path is not None:
                with open(save_input_output_to_path, 'w') as f:
                    json.dump(inputs_outputs, f, indent=2)

        # Write version sidecar once at the start of the run
        meta = {"version": get_version(), "driver": "langchain_agent"}
        if getattr(adm, 'architecture', None) is not None:
            meta['architecture'] = adm.architecture
        username = getattr(interface, 'username', None)
        if username is not None:
            meta["username"] = username
        with open(os.path.join(output_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        session_alignment_scores = []

        # Capture time it takes to choose each action
        action_times = {"scenarios": []}

        # Loop through available scenarios
        while scenario := interface.start_scenario():
            if scenario.id() == '':
                log.info("Next scenario ID is blank, assuming we're done, exiting")
                break
            alignment_target = self._get_alignment_target(cfg, scenario)
            agent_log_event(
                log, 'WORKFLOW', 'SCENARIO', 'Starting scenario',
                scenario_id=scenario.id(), target_id=getattr(alignment_target, 'id', None))
            if alignment_target is not None:
                if save_alignment_targets_to_path is not None:
                    alignment_target_path = os.path.join(
                        save_alignment_targets_to_path,
                        f"{alignment_target.id}.json")

                    with open(alignment_target_path, "w") as f:
                        json.dump(alignment_target.to_dict(), f, indent=2)

            session = ScenarioSession(
                scenario=scenario,
                alignment_target=(alignment_target
                                  if cfg.align_to_target else None),
                apply_action_filtering=self.apply_action_filtering,
                sort_available_actions=sort_available_actions,
                record_input_output=record_input_output,
                scene_end_policy=self.scene_end_policy,
                max_actions_per_scenario=self.max_actions_per_scenario,
                align_to_target=cfg.align_to_target)

            tools = self._build_tools(session)
            adm.run_scenario(session, tools, self.extra_tools)

            if session.scenario_complete:
                final_state = session.current_state
                agent_log_event(
                    log, 'ENVIRONMENT', 'FINAL STATE',
                    preview(final_state.unstructured), payload=final_state.unstructured,
                    **session.log_context())

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
                action_times["scenarios"].append(
                    _compute_time_stats(session.times_s))

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

                    agent_log_event(
                        log, 'ENVIRONMENT', 'ALIGNMENT SCORE',
                        preview(json.dumps(session_alignment_dict)),
                        payload=session_alignment_dict, **session.log_context())

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
