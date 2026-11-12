import inspect
import json
import logging
from pathlib import Path

from align_system.interfaces.cage_observations import (
    DEFENDER_FIELDS, capture_paths, format_window, trajectory_length, validate_fields,
)
from align_system.interfaces.abstracts import (
    Interface,
    ActionBasedScenarioInterface)

log = logging.getLogger(__name__)

RED_AGENT_LABELS = ['Sleep', 'Meander', 'B-line']

class CAGEActionBasedServiceInterface(Interface):
    seed = None
    cyborg_version = '2.1'
    scenario = 'base'
    def __init__(self,
            n_rollouts:int = 10,
            episode_length:int = 4,
            trajectory_dir:str = None,
            observation_fields = DEFENDER_FIELDS,
            red_agent_labels = None,
            start_step:int = 0,
            window_size:int = None,
            window_stride:int = 1,
                 ):
        self.n_rollouts = n_rollouts
        self.episode_length = episode_length
        self.current_rollout = 0
        self.wrapped_cyborg = None
        self.red_agent_labels = list(RED_AGENT_LABELS if red_agent_labels is None else red_agent_labels)
        # Zero-based first step included in trajectory replay observations.
        self.start_step = start_step
        self.window_size = window_size
        self.window_stride = window_stride

        # When set, replay trajectory JSONs exported by CAGE2's
        # test_experiment.py instead of simulating episodes
        self.trajectory_dir = None
        self.observation_fields = list(observation_fields)
        if trajectory_dir is not None:
            if type(start_step) is not int or start_step < 0:
                raise ValueError('start_step must be a non-negative integer')
            if type(episode_length) is not int or episode_length <= start_step:
                raise ValueError('episode_length must be an integer greater than start_step')
            if window_size is not None and (type(window_size) is not int or window_size < 1):
                raise ValueError('window_size must be a positive integer or null')
            if type(window_stride) is not int or window_stride < 1:
                raise ValueError('window_stride must be a positive integer')
            # Hydra changes the working directory, so resolve relative
            # paths against the directory the CLI was run from
            self.trajectory_dir = Path(trajectory_dir)
            if not self.trajectory_dir.is_absolute():
                from hydra.utils import to_absolute_path
                self.trajectory_dir = Path(to_absolute_path(trajectory_dir))
            validate_fields(self.observation_fields)

            trajectory_paths = (capture_paths(self.trajectory_dir)
                                if (self.trajectory_dir / 'summary.json').exists()
                                else sorted(self.trajectory_dir.rglob('*.json')))
            if not trajectory_paths:
                raise ValueError(f"No trajectory JSON files found under {self.trajectory_dir}")
            if n_rollouts > len(trajectory_paths):
                log.warning(f"n_rollouts={n_rollouts} but only {len(trajectory_paths)} "
                            f"trajectories found under {self.trajectory_dir}, replaying all of them")
            self.trajectory_paths = trajectory_paths[:n_rollouts]
            self._trajectory_scenarios = self._iter_trajectory_scenarios()

    def start_scenario(self):
        if self.trajectory_dir is not None:
            return self._start_trajectory_scenario()

        from CybORG import CybORG
        from CybORG.Agents.SimpleAgents.Meander_Resilience import RedMeanderAgent_Resilience
        from CybORG.Agents.Wrappers import BlueTableWrapper

        self.current_rollout += 1
        log.info("Starting CAGE Scenario") #f"*ADM Name*: {self.username}")
        if self.current_rollout > self.n_rollouts:
            log.info("Reached max # of CAGE rollouts")
            self.current_rollout = ""

        # TODO: we need to set up the CAGE environment here, and specify what agents are doing the scenario
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + '/Shared/Scenarios/scalability_experiments/base.yaml'
        # path = path[:-10] + '/Shared/Scenarios/Scenario2_resilience.yaml'
        # path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

        print(f'using CybORG v{self.cyborg_version}, {self.scenario}\n')

        # cyborg = CybORG(path, 'sim', agents={'Red': SleepAgent})
        cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent_Resilience})
        # cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent_Resilience})
        # cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent_Resilience_C})
        self.wrapped_cyborg = BlueTableWrapper(cyborg, output_mode = 'table')

        return CAGEActionBasedScenario(self.wrapped_cyborg, episode_length=self.episode_length,
                                      episode_number=self.current_rollout,
                                      red_agent_labels=self.red_agent_labels)

    def _start_trajectory_scenario(self):
        scenario = next(self._trajectory_scenarios, None)
        if scenario is None:
            log.info("Replayed all CAGE trajectories")
        return scenario

    def _iter_trajectory_scenarios(self):
        # n_rollouts counts input trajectories, not the windows emitted below.
        for trajectory_path in self.trajectory_paths:
            self.current_rollout += 1
            log.info(f"Replaying CAGE trajectory {trajectory_path}")
            with trajectory_path.open() as stream:
                trajectory = json.load(stream)
            trajectory_id = str(trajectory_path.relative_to(self.trajectory_dir))
            episode_end = min(self.episode_length, trajectory_length(trajectory, self.observation_fields))
            if self.window_size is None:
                windows = [(None, self.start_step, episode_end)]
            else:
                if episode_end - self.start_step < self.window_size:
                    raise ValueError(
                        f'No complete {self.window_size}-step window in '
                        f'[{self.start_step}, {episode_end}) for {trajectory_id}')
                windows = ((index, start, start + self.window_size)
                           for index, start in enumerate(range(
                               self.start_step, episode_end - self.window_size + 1,
                               self.window_stride)))
            for index, start, end in windows:
                yield CAGETrajectoryScenario(
                    trajectory_path, scenario_id=trajectory_id,
                    episode_length=end, observation_fields=self.observation_fields,
                    red_agent_labels=self.red_agent_labels, start_step=start,
                    trajectory_data=trajectory, window_index=index)

    def get_session_alignment(self, alignment_target):
        if self.wrapped_cyborg is not None:
            cia_scores = self.wrapped_cyborg.get_cia_scores()
            rewards = self.wrapped_cyborg.get_collected_rewards()
            return cia_scores | rewards

    def cli_parser(cls, parser=None):
        pass

    def cli_parser_description(cls):
        pass

    def init_from_parsed_args(cls, parsed_args):
        pass

class MetaInfo(object):
    def __init__(self, scene_id):
        self.scene_id = scene_id


#class CAGEAlignmentTarget(object):
#    def __init__(self, kdma_values):
#        self.kdma_values = kdma_values

class CAGEState:
    def __init__(self, table, hostnames, scene_id):
        self.meta_info = MetaInfo(scene_id)
        self.unstructured = str(table)
        self.hostnames = hostnames
        self.scenario_complete = False
        self.elapsed_time = 0

    

    def to_dict(self):
        return {'meta_info': {'scene_id':self.meta_info.scene_id},
                'unstructured': self.unstructured,
                'hostnames': self.hostnames,
                'scenario_complete': self.scenario_complete}

    def __repr__(self):
        return repr(self.to_dict())


class CAGEAction:
    def __init__(self, cage_class):
        self.cage_class = cage_class
        self.name = cage_class.__name__
        self.hostname = None
        self.justification = None
        self.unstructured = self.name
        self.kdma_association = None
        self.action_id = self.name.lower() 

    def to_dict(self):
        return {'name': self.name, 
                "hostname": self.hostname,
                "justification": self.justification,
                "unstructured": self.unstructured,
                "kdma_association": self.kdma_association,
                "action_id": self.action_id}
    
class RedAgentAction:
    def __init__(self, name):
        self.name = name
        self.justification = None
        # self.unstructured = self.name 

    def to_dict(self):
        return {'name': self.name,
                "justification": self.justification}



class CAGEActionBasedScenario(ActionBasedScenarioInterface):
    agent_name = 'Blue'
    def __init__(self, cyborg_sim, episode_length = 500, episode_number = 0,
                 red_agent_labels=None):
        from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric

        self.done = False
        self.red_agent_labels = list(RED_AGENT_LABELS if red_agent_labels is None else red_agent_labels)
        self.hostnames =[]
        self.episode_number = episode_number
        self.episode_length = episode_length
        self.scenario_count = 0

        self.cyborg_sim = cyborg_sim
        cage_obs = cyborg_sim.reset() #agent='blue_agent_4')
        cage_act_space = self.cyborg_sim.get_action_space(self.agent_name)

        self.hostnames = list(cage_act_space['hostname'].keys())
        self.obs = CAGEState(cage_obs.observation, self.hostnames, episode_number)
        self.reward = 0
        self.cia_scores = 0
        self.enrich_obs()
        self.metric = ResilienceMetric()
        #self.obs.scenario_complete = False

    def enrich_obs(self):
        self.obs.scenario_complete = self.done

    def id(self):
        return str(self.episode_number) 

    def get_alignment_target(self):
        ## This is defined via a configuration, not in here...
        pass

    def to_dict(self):
        pass
        #return self.scenario.__dict__

    def data(self):
        pass
        #return self.scenario

    def get_available_actions(self):
        cage_act_space = self.cyborg_sim.get_action_space(self.agent_name)
        # print(cage_act_space)
        return [CAGEAction(k) for k in cage_act_space['action']]
    
    def get_available_Red_actions(self):
        return [RedAgentAction(k) for k in self.red_agent_labels]

    def _take_or_intend_action(self, align_system_action):
        # Convert to proper 'Action' object prior to submission
        # align_system_action
        if align_system_action.name == "Sleep":
            action = align_system_action.cage_class()
        else:
            if align_system_action.hostname == None:
                action = align_system_action.cage_class(agent = "Blue", session = 0)
            else:
                action = align_system_action.cage_class(hostname = align_system_action.hostname, 
                                                agent = "Blue", session = 0)

        ## TODO takes an action and updates the state
        res = self.cyborg_sim.step(action=action, agent='Blue')

        self.scenario_count += 1
        cage_obs = res.observation
        self.reward = res.reward
        self.done = res.done or self.scenario_count >= self.episode_length 
        self.obs.unstructured = str(cage_obs)

        self.cia_scores = self.metric.calculate_scores(env = self.cyborg_sim, blue_action = action)
        return self.get_state()
    
    def step(self, action):
        ## TODO takes an action and updates the state
        res = self.cyborg_sim.step(action=action, agent='Blue')

        self.scenario_count += 1
        cage_obs = res.observation
        self.reward = res.reward
        self.done = res.done or self.scenario_count >= self.episode_length 
        self.obs.unstructured = str(cage_obs)

        self.cia_scores = self.metric.calculate_scores(env = self.cyborg_sim, blue_action = action)
        return self.get_state()

    def intend_action(self, action):
        return self._take_or_intend_action(
            action,
        )

    def take_action(self, action):
        return self._take_or_intend_action(
            action, 
        )

    def get_state(self):
        #text_state = format_observation(self.obs, self.last_action, BLUE_AGENT_NAME)
        ## convert the state into a string for the LLM
        self.enrich_obs()
        return self.obs #swagger_state


class CAGETrajectoryScenario(ActionBasedScenarioInterface):
    # Replays a trajectory JSON exported by CAGE2's AgentTrajectory.
    # Nothing is simulated, so the scenario starts complete with the
    # formatted trajectory as its state
    def __init__(self, trajectory_path, scenario_id, episode_length, observation_fields,
                 red_agent_labels=None, start_step=0, trajectory_data=None, window_index=None):
        if trajectory_data is None:
            with open(trajectory_path) as f:
                trajectory_data = json.load(f)
        self.trajectory = trajectory_data
        self.trajectory_id = scenario_id
        self.window_index = window_index
        self.window_start = start_step if window_index is not None else None
        # Exclusive end: [10, 15) contains steps 10, 11, 12, 13, 14.
        self.window_end = episode_length if window_index is not None else None
        self.window_size = episode_length - start_step if window_index is not None else None
        self.scenario_id = (f'{scenario_id}#steps={start_step}:{episode_length}'
                            if window_index is not None else scenario_id)
        self.red_agent_name = self.trajectory['red_agent_name']
        self.red_agent_labels = list(RED_AGENT_LABELS if red_agent_labels is None else red_agent_labels)
        if not isinstance(start_step, int) or start_step < 0:
            raise ValueError('start_step must be a non-negative integer')
        self.start_step = start_step

        self.obs = CAGEState(self.format_trajectory(episode_length, observation_fields),
                             list(self.trajectory.get('network_topology', {}).keys()),
                             self.scenario_id)
        self.obs.scenario_complete = True

    def format_trajectory(self, episode_length, observation_fields):
        return format_window(self.trajectory, self.start_step, episode_length, observation_fields)

    def get_red_agent_label(self):
        # Preserve family-only scoring for the default classifier, and use
        # C/I/A subclasses when those labels are enabled for resilience runs.
        if 'B_line' in self.red_agent_name or self.red_agent_name.split('_')[0] == 'BlineJumping':
            family = 'B-line'
        elif 'Meander' in self.red_agent_name:
            family = 'Meander'
        elif 'Sleep' in self.red_agent_name:
            family = 'Sleep'
        else:
            return None

        # Match whole tokens: Combination_CIA must not be mistaken for C.
        for dimension in ('C', 'I', 'A'):
            label = f'{family}-{dimension}'
            if dimension in self.red_agent_name.split('_') and label in self.red_agent_labels:
                return label
        return family if family in self.red_agent_labels else None

    def id(self):
        return self.scenario_id

    def get_alignment_target(self):
        pass

    def to_dict(self):
        return self.trajectory

    def data(self):
        return self.trajectory

    def get_available_actions(self):
        return []

    def get_available_Red_actions(self):
        return [RedAgentAction(k) for k in self.red_agent_labels]

    def intend_action(self, action):
        raise RuntimeError("Trajectory replays can't take actions")

    def take_action(self, action):
        raise RuntimeError("Trajectory replays can't take actions")

    def get_state(self):
        return self.obs
