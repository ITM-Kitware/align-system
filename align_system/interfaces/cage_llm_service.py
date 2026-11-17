"""Seeded, live CAGE2 episodes for an LLM-only Blue defender."""

from copy import copy
from pathlib import Path
import random

import numpy as np
from hydra.utils import to_absolute_path
from CybORG import CybORG
from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Agents.SimpleAgents.B_line_resilience import (
    BlineJumping_C, BlineJumping_I, BlineJumping_A,
)
from CybORG.Agents.SimpleAgents.Meander_Resilience import (
    MeanderJumping_C, MeanderJumping_I, MeanderJumping_A,
)

from align_system.interfaces.cia_cage2_service import (
    CAGEActionBasedServiceInterface, CAGEActionBasedScenario,
)


ATTACKERS = {
    'bline_c': BlineJumping_C, 'bline_i': BlineJumping_I, 'bline_a': BlineJumping_A,
    'meander_c': MeanderJumping_C, 'meander_i': MeanderJumping_I,
    'meander_a': MeanderJumping_A,
}


class CAGELLMServiceInterface(CAGEActionBasedServiceInterface):
    def __init__(self, scenario_path, red_agent='bline_c', n_rollouts=100,
                 episode_length=100, start_seed=1000):
        for name, value, minimum in (
            ('n_rollouts', n_rollouts, 1), ('episode_length', episode_length, 1),
            ('start_seed', start_seed, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if start_seed + n_rollouts > 2**32:
            raise ValueError('Episode seeds must fit in a NumPy uint32 seed')
        if red_agent not in ATTACKERS:
            raise ValueError(f'red_agent must be one of {list(ATTACKERS)}')
        self.scenario_path = Path(to_absolute_path(scenario_path))
        if not self.scenario_path.is_file():
            raise FileNotFoundError(self.scenario_path)
        super().__init__(n_rollouts=n_rollouts, episode_length=episode_length)
        self.red_agent = red_agent
        self.start_seed = start_seed

    def start_scenario(self):
        if self.current_rollout >= self.n_rollouts:
            return None
        seed = self.start_seed + self.current_rollout
        # CAGE2 and its jumping attackers use process-global Python/NumPy RNGs.
        # Seed before construction as well as the reset performed by Scenario.
        random.seed(seed)
        np.random.seed(seed)
        cyborg = CybORG(str(self.scenario_path), 'sim',
                        agents={'Red': ATTACKERS[self.red_agent]})
        self.wrapped_cyborg = BlueTableWrapper(cyborg, output_mode='table')
        # Match CAGE2's evaluation convention: reset starts at this exact seed,
        # independent of random draws during simulator construction.
        cyborg.set_seed(seed)
        np.random.seed(seed)
        scenario = CAGELLMScenario(self.wrapped_cyborg, self.episode_length, seed)
        scenario.seed = seed
        self.current_rollout += 1
        return scenario


class CAGELLMScenario(CAGEActionBasedScenario):
    def get_state(self):
        # Render Blue's own detections in the existing ACD prompt's format.
        # Never expose the Red actions or simulator ground truth to the LLM.
        state = copy(super().get_state())
        alerts = []
        for _, _, hostname, activity, compromised in self.cyborg_sim.info.values():
            if activity in ('Scan', 'Exploit'):
                alerts.append(f'{hostname}: WARNING: A suspicious connection has been detected.')
            if compromised == 'User':
                alerts.append(f'{hostname}: ALERT: A red agent has been detected on the host.')
            elif compromised == 'Privileged':
                alerts.append(f'{hostname}: CRITICAL: A red agent has been detected with a privileged escalation.')
        last_action = self.cyborg_sim.get_last_action('Blue')
        blue_obs = self.cyborg_sim.get_attr('get_observation')('Blue')
        state.unstructured = (
            '# CURRENT SCENARIO\n'
            'Use these actual hostnames instead of the example HOSTNAMES in the system prompt:\n'
            + ', '.join(self.hostnames)
            + '\nAuth0, Database0, and Front0 are the critical CIA service hosts. '
            'The available Decoy* actions are concrete variants of DecoyAction; each requires a hostname. '
            'Sleep takes no hostname and does nothing. Select only an action from Responses.\n\n'
            '# OBSERVATION\n'
            f'Last Action: {last_action}\n'
            f'Last Action Status: {blue_obs.get("success", "UNKNOWN")}\n'
            'Suspicious Activity Detected:\n'
            + ('\n'.join(alerts) if alerts else 'None')
        )
        state.elapsed_time = self.scenario_count
        return state
