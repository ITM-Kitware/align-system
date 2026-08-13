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

import pprint
import numpy as np
from cyborg.env import cyborg as CybORG
from cyborg.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from cyborg.Simulator.Scenarios import EnterpriseScenarioGenerator
from cyborg.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE
# from cyborg.Evaluation.llamagym.submission import PhaseWrapper
# from cyborgm.Evaluation.example_submission.submission import Submission
log = logging.getLogger(__name__)
from cyborg.Agents.LLMAgents.llm_adapter.obs_formatter import _format_suspicious_activity
from cyborg.Agents.HMARLAgents.submission import Submission

import os
import sys

class CAGEActionBasedServiceInterface(Interface):
    EPISODE_LENGTH=100
    seed = None
    cyborg_version = '2.1'
    scenario = 'base'
    def __init__(self,
            n_rollouts:int = 1,
                 ):
        self.n_rollouts = n_rollouts
        self.current_rollout = 0
        self.wrapped_cyborg = None
        self.red_agent = None
        self.rng = np.random.default_rng()

    def start_scenario(self):
        self.current_rollout += 1
        # log.info("Starting CAGE Scenario") #f"*ADM Name*: {self.username}")
        if self.current_rollout > self.n_rollouts:
            log.info("Reached max # of CAGE rollouts")
            self.current_rollout = ""

        sg = EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=EnterpriseGreenAgent,
                red_agent_class=FiniteStateRedAgent,
                steps=500,
        )
        cyborg = CybORG(sg, "sim")
        # self.wrapped_cyborg = EnterpriseMAE(cyborg)
        # self.wrapped_cyborg = PhaseWrapper(cyborg)
        # self.wrapped_cyborg = cyborg
        self.wrapped_cyborg = Submission.wrap(cyborg)

        return CAGEActionBasedScenario(self.wrapped_cyborg, episode_length=self.EPISODE_LENGTH, episode_number = self.current_rollout)

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
    
class MetaAgentAction:
    def __init__(self, name):
        self.name = name
        self.justification = None
        # self.unstructured = self.name 

    def to_dict(self):
        return {'name': self.name,
                "justification": self.justification}



class CAGEActionBasedScenario(ActionBasedScenarioInterface):
    agent_name = 'Blue'
    def __init__(self, cyborg_sim, episode_length = 500, episode_number = 0):
        self.done = False
        self.hostnames =[]
        self.episode_number = episode_number
        self.episode_length = episode_length
        self.scenario_count = 0

        self.cyborg_sim = cyborg_sim
        self.observation, _ = cyborg_sim.reset()

        self.agents = ['blue_agent_0', 'blue_agent_1', 'blue_agent_2', 'blue_agent_3', 'blue_agent_4']

        cage_obs = "Observation:\n"
        for agent in self.agents:
            cage_obs += "{0}:\n".format(agent)
            # s = self.cyborg_sim.get_observation(agent)
            s = self.cyborg_sim.env.get_observation(agent)
            # print(s)
            # exit()
            s = _format_suspicious_activity(s)
            if len(s) == 0:
                cage_obs += "None\n"
            else:
                for i in s:
                    cage_obs += "{0}\n".format(i)

        self.obs = CAGEState(cage_obs, self.hostnames, episode_number)
        self.reward = 0
        self.cia_scores = 0
        self.metric = None
        self.obs.scenario_complete = False
        self.info = None

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
        return [CAGEAction(k) for k in cage_act_space['action']]
    
    def get_available_meta_actions(self):
        available_actions = ['Investigate', 'Recover']
        return [MetaAgentAction(k) for k in available_actions]

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

        res = self.cyborg_sim.step(action=action, agent='Blue')

        self.scenario_count += 1
        cage_obs = res.observation
        self.reward = res.reward
        self.done = res.done or self.scenario_count >= self.episode_length 
        self.obs.unstructured = str(cage_obs)

        self.cia_scores = self.metric.calculate_scores(env = self.cyborg_sim, blue_action = action)
        return self.get_state()
    
    def step(self, action):
        self.observation, rew, term, trunc, self.info = self.cyborg_sim.step(action)
        # print(self.observation)
        # exit()
        # pprint.pprint(r)
        cage_obs = "Observation:\n"
        for agent in self.agents:
            cage_obs += "{0}:\n".format(agent)
            # s = self.cyborg_sim.get_observation(agent)
            s = self.cyborg_sim.env.get_observation(agent)
            s = _format_suspicious_activity(s)
            # pprint.pprint(x)
            if len(s) == 0:
                cage_obs += "None\n"
            else:
                for i in s:
                    cage_obs += "{0}\n".format(i)
        # print(cage_obs)
        self.scenario_count += 1
        # self.reward = rewards
        self.done = False or self.scenario_count >= self.episode_length 
        self.obs.unstructured = str(cage_obs)
        self.reward = list(rew.values())[0]
        # self.cia_scores = self.metric.calculate_scores(env = self.cyborg_sim, blue_action = action)
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

