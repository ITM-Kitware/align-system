import inspect
from pathlib import Path

from hydra.utils import to_absolute_path

from align_system.utils import logging
from align_system.interfaces.abstracts import (
    Interface,
    ActionBasedScenarioInterface)

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.B_line_resilience import (
    B_lineAgent_Resilience,
    B_lineAgent_Resilience_C_multiTarget,
    B_lineAgent_Resilience_I_multiTarget,
    B_lineAgent_Resilience_A_multiTarget,
)
from CybORG.Agents.SimpleAgents.SleepAgent import SleepAgent
from CybORG.Agents.SimpleAgents.Meander_Resilience import (
    RedMeanderAgent_Resilience,
    RedMeanderAgent_Combination_multiTarget_C,
    RedMeanderAgent_Combination_multiTarget_I,
    RedMeanderAgent_Combination_multiTarget_A,
)
from CybORG.Agents.Wrappers import BlueTableWrapper

from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric

log = logging.getLogger(__name__)

RED_AGENTS = {
    'Sleep': SleepAgent,
    'Meander': RedMeanderAgent_Resilience,
    'B-line': B_lineAgent_Resilience,
    'B-line-C': B_lineAgent_Resilience_C_multiTarget,
    'B-line-I': B_lineAgent_Resilience_I_multiTarget,
    'B-line-A': B_lineAgent_Resilience_A_multiTarget,
    'Meander-C': RedMeanderAgent_Combination_multiTarget_C,
    'Meander-I': RedMeanderAgent_Combination_multiTarget_I,
    'Meander-A': RedMeanderAgent_Combination_multiTarget_A,
}


class CAGEActionBasedServiceInterface(Interface):
    EPISODE_LENGTH=100
    seed = None
    cyborg_version = '2.1'
    scenario = 'base'
    # scenario = 'Scenario1b'
    def __init__(self,
            n_rollouts:int = 100,
            scenario_path:str = None,
            red_agent:str = 'Meander',
            red_agent_labels=None,
                 ):
        self.n_rollouts = n_rollouts
        self.current_rollout = 0
        self.wrapped_cyborg = None
        self.red_agent_labels = list(RED_AGENTS if red_agent_labels is None else red_agent_labels)
        unknown_labels = set(self.red_agent_labels) - RED_AGENTS.keys()
        if unknown_labels or not self.red_agent_labels:
            raise ValueError(f"red_agent_labels must be a non-empty subset of {list(RED_AGENTS)}")
        if red_agent not in self.red_agent_labels:
            raise ValueError(f"red_agent {red_agent!r} must be one of red_agent_labels")
        self.red_agent_label = red_agent
        self.red_agent = RED_AGENTS[red_agent]
        if scenario_path is None:
            self.scenario_path = (
                Path(inspect.getfile(CybORG)).parent
                / 'Shared/Scenarios/scalability_experiments/base.yaml'
            )
        else:
            # Resolve against Hydra's original working directory, not its run directory.
            self.scenario_path = Path(to_absolute_path(scenario_path))
        if not self.scenario_path.is_file():
            raise FileNotFoundError(f"CAGE2 scenario not found: {self.scenario_path}")

    def start_scenario(self):
        if self.current_rollout >= self.n_rollouts:
            log.info("Reached max # of CAGE rollouts")
            return None
        self.current_rollout += 1

        cyborg = CybORG(str(self.scenario_path), 'sim', agents={'Red': self.red_agent})
        self.wrapped_cyborg = BlueTableWrapper(cyborg, output_mode = 'table')

        return CAGEActionBasedScenario(
            self.wrapped_cyborg, episode_length=self.EPISODE_LENGTH,
            episode_number=self.current_rollout, red_agent_labels=self.red_agent_labels)

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
#    def __init__(self, kdma_values):observation
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
        self.done = False
        self.red_agent_labels = list(RED_AGENTS if red_agent_labels is None else red_agent_labels)
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

    def get_policy_observation(self):
        # Match the BlueTableWrapper encoding used by ChallengeWrapper during PPO training.
        return self.cyborg_sim._create_vector(success=None)

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

