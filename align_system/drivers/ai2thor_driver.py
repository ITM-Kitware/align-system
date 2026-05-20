from __future__ import annotations

import json
import os
from timeit import default_timer as timer

import hydra
from omegaconf import DictConfig, OmegaConf

from align_system.interfaces.ai2thor_interface import AI2ThorAction
from align_system.data_models.types import Action as MCTSAction
from align_system.utils import logging

log = logging.getLogger(__name__)


class AI2ThorDriver:
    """
    Drives the MCTS agent through AI2Thor scenarios using the same
    interface/ADM/driver pattern as ITMPhase1Driver.
    """

    def __init__(self, max_steps: int = 40, verbose: bool = True):
        self.max_steps = max_steps
        self.verbose = verbose

    def drive(self, cfg: DictConfig) -> None:
        interface = cfg.interface
        adm = cfg.adm.instance

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # Route frames into the Hydra output dir so each run is self-contained
        if getattr(interface, "save_frames", False):
            interface.frame_dir = os.path.join(output_dir, "frames")

        save_input_output_to_path = None
        if cfg.get("save_input_output", False):
            save_input_output_to_path = os.path.join(output_dir, "input_output.json")

        save_timing_to_path = None
        if cfg.get("save_timing", False):
            save_timing_to_path = os.path.join(output_dir, "timing.json")

        inputs_outputs = []
        timing = {"scenarios": []}

        align_to_target = cfg.get("align_to_target", False)
        if 'alignment_target' in cfg and align_to_target:
            alignment_target = cfg.alignment_target
            alignment_target.kdma_values = [OmegaConf.to_container(c)
                                            if isinstance(c, DictConfig) else c
                                            for c in alignment_target.kdma_values]
        else:
            alignment_target = None

        while scenario := interface.start_scenario():
            log.info(f"[bold]*Scenario*[/bold]: {scenario.id()}", extra={"markup": True})

            if hasattr(adm, "reset_history"):
                adm.reset_history()

            current_state = scenario.get_state()
            available_actions = scenario.get_available_actions()

            step = 0
            sce_times = []

            while not current_state.scenario_complete and step < self.max_steps:
                if self.verbose:
                    log.info(f"[t={step}] obs: {current_state.unstructured[:120]}...")

                start = timer()
                choose_result = adm.choose_action(
                    current_state,
                    available_actions,
                    alignment_target=alignment_target,
                    scenario_id=scenario.id(),
                )
                sce_times.append(timer() - start)

                if isinstance(choose_result, tuple):
                    action_to_take, choice_info = choose_result
                else:
                    action_to_take, choice_info = choose_result, {}

                plan = getattr(action_to_take, "plan", None) or [None]
                executed_plan: list[MCTSAction] = []

                for plan_idx, plan_action in enumerate(plan):
                    if plan_action is None:
                        # No plan attached; execute the top-level action directly
                        exec_action = action_to_take
                    else:
                        exec_action = AI2ThorAction(
                            action_id=plan_action.tool_name,
                            unstructured=action_to_take.unstructured,
                            args=plan_action.args or {},
                        )

                    log.info(f"[t={step}] action: {exec_action.action_id}")

                    inputs_outputs.append({
                        "scenario_id": scenario.id(),
                        "step": step,
                        "state": current_state.unstructured,
                        "action": exec_action.to_dict(),
                    })

                    if save_input_output_to_path is not None:
                        with open(save_input_output_to_path, "w") as f:
                            json.dump(inputs_outputs, f, indent=2)

                    prev_env_step = getattr(current_state, "env_step", -1)
                    current_state = scenario.take_action(exec_action)
                    step += 1

                    if getattr(current_state, "env_step", -1) != prev_env_step:
                        executed_plan.append(
                            plan_action if plan_action is not None
                            else MCTSAction(tool_name=exec_action.action_id, args=exec_action.args or {})
                        )
                    else:
                        log.info(f"[t={step-1}] action {exec_action.action_id} had no effect (env_step unchanged); skipping history")

                    if current_state.scenario_complete:
                        log.info(f"[bold]Task complete after {step} steps.[/bold]",
                                 extra={"markup": True})
                        break

                if executed_plan and hasattr(adm, "update_history"):
                    adm.update_history(
                        AI2ThorAction(
                            action_id=executed_plan[0].tool_name,
                            unstructured=action_to_take.unstructured,
                            plan=executed_plan,
                        )
                    )

            if step >= self.max_steps and not current_state.scenario_complete:
                log.warning(f"Reached max_steps ({self.max_steps}) without completing task.")

            timing["scenarios"].append({
                "scenario_id": scenario.id(),
                "n_steps": step,
                "total_time_s": sum(sce_times),
                "avg_time_s": sum(sce_times) / len(sce_times) if sce_times else 0,
            })

        if save_timing_to_path is not None:
            with open(save_timing_to_path, "w") as f:
                json.dump(timing, f, indent=2)
