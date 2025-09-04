# Test file to run CybORG challenge 1


from CybORG.Shared.Enums import TrinaryEnum
from pprint import pprint

# Calculates CIA attributes given a CAGE simulation environment step
class CIATriadMetric:
    
    def __init__(self,
            user_host_w:float=0.34,
            enterprise_host_w:float=0.34,
            operational_host_w:float=0.34,
            availability_multiplier:float=1.5,
            restoration_weight:float=0.5
            ):
        
        #weights assigned to different host types
        self.user_host_w = user_host_w
        self.enterprise_host_w = enterprise_host_w
        self.operational_host_w = operational_host_w
        
        #multiplier for availability. Each step a system is down, availability
        # is impacted x multipler
        self.availability_multiplier = availability_multiplier
        self.restoration_weight = restoration_weight
        
        #holds CIA over each step of the game
        self.confidentialities = []
        self.integrities = []
        self.availabilities = []
        
        self.network_topology = {} #holds information about each host on the network over time
        
        
    def calculate_cia(self, env):
        """Calculate CIA attributes from cyborg observation table. 

        Args:
            env (CybORG): CybORG class environment with dictionary-like observations
        """

        confidentiality, integrity, availability = 0,0,0
        
        # Blue agent action posibilities: ['Sleep', 'Monitor', 'Analyse', 'Remove', 'Restore']
        blue_agent_env_action = env.get_last_action("Blue")
        blue_agent_action = blue_agent_env_action.__class__.__name__
        blue_agent_obs = env.get_observation("Blue")

        red_agent_env_action = env.get_last_action("Red")
        red_agent_action = red_agent_env_action.__class__.__name__
        red_agent_obs = env.get_observation("Red")

        
        # Confidentiality: 
        # -1.0 x host type if red agent successfully exploits a host
        # 1.0 x host type if blue agent successfully removes red agent access to a host
        if red_agent_action == "ExploitRemoteService" and red_agent_obs["success"] == TrinaryEnum.TRUE:
            exploited_host = red_agent_obs[str(red_agent_env_action.ip_address)]["System info"]["Hostname"]
            self.network_topology[exploited_host]["Confidentiality"] = -1.0
        if blue_agent_action == "Remove" and blue_agent_obs["success"] == TrinaryEnum.TRUE:
            # only count as an improvement in confidentiality if red agent had a shell open in the host
            if blue_agent_env_action.hostname in self.network_topology and self.network_topology[blue_agent_env_action.hostname]["Confidentiality"] < 0:
                self.network_topology[blue_agent_env_action.hostname]["Confidentiality"] = 1.0

        # calculate confidentiality
        for sys, details in self.network_topology.items():
            #weigh system based on host type
            confidentiality += self.network_topology[sys]["Confidentiality"]*self.get_sys_weight(sys)
            
            
        # Integrity: 
        # -1.0 x host type if red agent successfully gets root access on a host
        # 1.0 x host type if blue agent successfully removes root access of a red agent
        #       or if red agent's root access fails (was removed in a prior step)
        if red_agent_action == "PrivilegeEscalate":
            if red_agent_obs["success"] == TrinaryEnum.TRUE: 
                self.network_topology[red_agent_env_action.hostname]["Integrity"] = -1.0
                
        # calculate integrity
        for sys, details in self.network_topology.items():
            #weigh system based on host type
            integrity += self.network_topology[sys]["Integrity"]*self.get_sys_weight(sys)
            
            
        # Availability: 
        # 1.0 x host type if service is available
        # -1.0 x host type x availability_multiplier if service is not available
        #       and will be set to 1.0 if blue agent performs a "Restore"
        if red_agent_action == "Impact" and red_agent_obs["success"] == TrinaryEnum.TRUE:
            #if this was already impacted, apply multiplier
            if self.network_topology[red_agent_env_action.hostname]["Available"] < 0:
                self.network_topology[red_agent_env_action.hostname]["Available"] *= self.availability_multiplier
            else:
                self.network_topology[red_agent_env_action.hostname]["Available"] = -1.0
        if blue_agent_action == "Restore" and blue_agent_obs["success"] == TrinaryEnum.TRUE and \
            blue_agent_env_action.hostname in self.network_topology and self.network_topology[blue_agent_env_action.hostname]["Available"] < 0:
            # doing restore resets the host to its initial state, so everything is restored
            # causes disruption for next step though - reward reflects
            self.network_topology[blue_agent_env_action.hostname]["Available"] = 1.0
            self.network_topology[blue_agent_env_action.hostname]["Confidentiality"] = 1.0
            self.network_topology[blue_agent_env_action.hostname]["Integrity"] = 1.0*self.restoration_weight


        # calculate availability
        for sys, details in self.network_topology.items():
            #weigh system based on host type
            availability += self.network_topology[sys]["Available"]*self.get_sys_weight(sys)
            
        self.confidentialities.append(confidentiality)
        self.integrities.append(integrity)
        self.availabilities.append(availability)
        
        return confidentiality, integrity, availability
    
    def get_sys_weight(self, system_name):
        # Calculate weight based on system_name string
        sys_w = self.user_host_w
        if "Enterprise" in system_name:
            sys_w = self.enterprise_host_w
        elif "Op" in system_name:
            sys_w = self.operational_host_w
        return sys_w
    
    def reset(self, network_observation):
        # Reset network topology for referencing later
        # and CIA scores
        self.set_network_topology(network_observation)
        self.confidentialities = []
        self.integrities = []
        self.availabilities = []
        
        
    def set_network_topology(self, network_observation):
        # Set all the hosts and subnets based on intiial network observation contianing all
        # CIA computation is based on this established network topology
        self.network_topology = {}
        for system, details in network_observation.items():
            if system == "Defender": 
                continue
            
            self.network_topology[system] = {
                "Confidentiality": 1,
                "Available": 1,
                "Integrity": 1
            }
    
    def scores(self):
        return {
            "C": sum(self.confidentialities),
            "I": sum(self.integrities),
            "A": sum(self.availabilities)
        }

    def __str__(self):
        return str(self.scores())
    

#
# Example:
# metric = CIATriadMetric()
# 
# Initialize network topology: 
# metric.set_network_topology(env.get_action_space("Blue"))
#
# Calculate CIA for one step:
# metric.calculate_cia(env)
#
# After finishing run to get final CIA scores:
# metric.scores()
