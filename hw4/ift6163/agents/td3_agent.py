import numpy as np

from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.policies.MLP_policy import MLPPolicyDeterministic
from ift6163.critics.td3_critic import TD3Critic
import copy

from ift6163.agents.ddpg_agent import DDPGAgent

class TD3Agent(DDPGAgent):
    def __init__(self, env, agent_params):

        super().__init__(env, agent_params)
        
        self.q_fun = TD3Critic(self.actor, 
                               agent_params, 
                               self.optimizer_spec, env)

        # for the Delayed part of Twin Delayed DDPG (TD3)
        self.policy_delay = 2 if agent_params['policy_delay'] == -1 else self.policy_delay
        