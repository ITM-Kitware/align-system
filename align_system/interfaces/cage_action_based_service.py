import argparse 
from uuid import uuid4
import inspect


from align_system.utils import logging
from align_system.interfaces.abstracts import (
    Interface,
    ActionBasedScenarioInterface)

from swagger_client.models import (
    State,
    Action,
    Character,
    Supplies,
    Injury,
    Environment,
    DecisionEnvironment,
    Aid,
    SimEnvironment, MetaInfo,
)

#import cyborg
#cyborg.Agents.LLMAgents.config.config_vars.NO_LLM_AGENTS=True
#from cyborg.Agents.LLMAgents.config.config_vars import BLUE_AGENT_NAME
#from cyborg.Agents.LLMAgents.llm_adapter.obs_formatter import format_observation
#from cyborg import cyborg, CYBORG_VERSION
#from cyborg.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
#from cyborg.Simulator.Scenarios import EnterpriseScenarioGenerator
#from align_system.interfaces.submission import Submission
#from cyborg.Evaluation.Cybermonics.submission import Submission

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import BlueTableWrapper


log = logging.getLogger(__name__)


class CAGEActionBasedServiceInterface(Interface):
    EPISODE_LENGTH=2
    seed = None
    cyborg_version = '1.2'
    scenario = 'Scenario1b'
    def __init__(self,
            n_rollouts:int = 1,
                 ):
        self.n_rollouts = n_rollouts
        self.current_rollout = 0

    def start_scenario(self):
        self.current_rollout += 1
        log.info("Starting CAGE Scenario") #f"*ADM Name*: {self.username}")
        if self.current_rollout > self.n_rollouts:
            log.info("Reached max # of CAGE rollouts")
            self.current_rollout = ""

        # TODO: we need to set up the CAGE environment here, and specify what agents are doing the scenario
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

        print(f'using CybORG v{self.cyborg_version}, {self.scenario}\n')

        cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
        wrapped_cyborg = BlueTableWrapper(cyborg, output_mode = 'table') #'blue_table')


        return CAGEActionBasedScenario(wrapped_cyborg, episode_length=self.EPISODE_LENGTH, episode_number = self.current_rollout)

    def get_session_alignment(self, alignment_target):
        return None
        if 0:
            if self.training_session == 'full':
                # 'solo' training sessions are not able to retrieve an
                # alignment score
                return self.connection.get_session_alignment(
                    self.session_id, alignment_target.id)
            else:
                return None

    def cli_parser(cls, parser=None):
        pass

    def cli_parser_description(cls):
        pass

    def init_from_parsed_args(cls, parsed_args):
        pass

class MetaInfo(object):
    def __init__(self, scene_id):
        self.scene_id = scene_id

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
        return {'name': self.name}



class CAGEActionBasedScenario(ActionBasedScenarioInterface):
    agent_name = 'Blue'
    def __init__(self, cyborg_sim, episode_length = 500, episode_number = 0):
        self.done = False
        self.hostnames =[]
        self.episode_number = episode_number
        self.episode_length = episode_length
        self.scenario_count = 0

        self.cyborg_sim = cyborg_sim
        cage_obs = cyborg_sim.reset() #agent='blue_agent_4')
        cage_act_space = self.cyborg_sim.get_action_space(self.agent_name)
        self.hostnames = list(cage_act_space['hostname'].keys())
        self.obs = CAGEState(cage_obs, self.hostnames, episode_number)
        self.enrich_obs()
        #self.obs.scenario_complete = False

    def enrich_obs(self):
        self.obs.scenario_complete = self.done
#        self.obs.meta_info = MetaInfo()
#        self.obs.hostnames = self.hostnames
#        setattr(self.obs.meta_info, 'scene_id', self.episode_number)

    def id(self):
        return str(self.episode_number) 

    def get_alignment_target(self):
        ## TODO: not sure what to do here...
        return self.connection.get_alignment_target(
            self.session_id, self.scenario.id)

    def to_dict(self):
        pass
        #return self.scenario.__dict__

    def data(self):
        pass
        #return self.scenario

    def get_available_actions(self):
        ## TODO: get the action space in the format that align expects
        cage_act_space = self.cyborg_sim.get_action_space(self.agent_name)
        return [CAGEAction(k) for k in cage_act_space['action']]

    def _take_or_intend_action(self, align_system_action):
        # Convert to proper 'Action' object prior to submission
        if align_system_action.hostname is None:
            action = align_system_action.cage_class(agent = "Blue", session = 0)
        else:
            action = align_system_action.cage_class(hostname = align_system_action.hostname, 
                                            agent = "Blue", session = 0)

        ## TODO takes an action and updates the state
        res = self.cyborg_sim.step(action=action, agent='Blue')
        self.scenario_count += 1
        cage_obs = res.observation
        rew = res.reward
        self.done = res.done or self.scenario_count >= self.episode_length 
        self.obs.unstructured = str(cage_obs)
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

def convert_action_space( action_space:dict):
    assert type(action_space) is dict, \
        f"Wrapper required a dictionary action space. " \
        f"Please check that the wrappers below the ReduceActionSpaceWrapper return the action space as a dict "
    possible_actions = []
    temp = {}
    params = ['action']
    # for action in action_space['action']:
    action_signature = {}
    for i, action in enumerate(action_space['action']):
        if action not in action_signature:
            action_signature[action] = inspect.signature(action).parameters
        param_dict = {}
        param_list = [{}]
        print(action)
        print(action_signature[action])
        for p in action_signature[action]:
            temp[p] = []
            if p not in params:
                params.append(p)

            print(action_space[p])

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
            print(p_dict)
            possible_actions.append(action(**p_dict))
    quit()
    return possible_actions

if __name__ == '__main__':
    intf = CAGEActionBasedServiceInterface()
    scen = intf.start_scenario()
    state = scen.get_state()
    print(state)
    action_space = scen.get_available_actions()
    print(action_space)
    quit()
    converted_actions = convert_action_space(action_space)
    print("==========================+=")
    print(converted_actions)
    quit()
    for aidx, action in enumerate(action_space['action']):
        print(dir(action))
#        print(action.get_params())
        align_action = Action(
                action_id = f'action-{aidx}',
                action_type = 'APPLY_TREATMENT',
                intent_action=True,
                unstructured=action.__name__,
                character_id = 'n/a',
                threat_state = None,
                parameters = {},
                justification = None,
                )
        #cage_action = 
        scen.take_action(align_action)
