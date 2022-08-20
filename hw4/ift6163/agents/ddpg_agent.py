import numpy as np
import torch

from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from ift6163.policies.MLP_policy import MLPPolicyDeterministic
from ift6163.critics.ddpg_critic import DDPGCritic
import copy
from ift6163.infrastructure import pytorch_util as ptu


class DDPGAgent(object):
    def __init__(self, env, agent_params):

        self.env = env

        self.agent_params = agent_params
        print ("agent_params", agent_params)
        self.batch_size = agent_params['train_batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()

        self.num_actions = self.env.action_space.shape[0]
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']
        self.policy_delay = 1 if agent_params['policy_delay'] == -1 else agent_params['policy_delay']
        self.exploration_noise = agent_params['exploration_noise']
        self.action_min = ptu.from_numpy(env.action_space.low)
        self.action_max = ptu.from_numpy(env.action_space.high)
        print(f"self.action_min={self.action_min}, self.action_max={self.action_max}")


        self.replay_buffer_idx = None
        self.optimizer_spec = agent_params['optimizer_spec']
        
        self.actor = MLPPolicyDeterministic(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=False,
            amplitude_action=env.action_space.high,
            activation=agent_params['activation'],
        )
        print("Actor")
        print(self.actor)
        # Create the Q function
        self.q_fun = DDPGCritic(self.actor, agent_params, self.optimizer_spec)

        # Hint: We can use the Memory optimized replay buffer but now we have continuous actions
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=True,
            continuous_actions=True, ac_dim=self.agent_params['ac_dim'])
        self.t = 0
        self.num_param_updates = 0
        
    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # DONE : store the latest observation ("frame") into the replay buffer
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        # DONE : add noise to the deterministic policy
        perform_random_action = self.t < self.learning_starts
        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            observation = self.replay_buffer.encode_recent_observation()
            observation = ptu.from_numpy(observation)
            action = self.actor(observation)  # deterministic action
            action = action + torch.randn_like(action) * self.exploration_noise # add exploration noise
            action = action.clamp(self.action_min, self.action_max)  # clip the action to what is allowed
            action = ptu.to_numpy(action)
        
        # DONE : take a step in the environment using the action from the policy
        self.last_obs, reward, done, info = self.env.step(action)

        # DONE : store the result of taking this action into the replay buffer
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # DONE : if taking this step resulted in done, reset the env and the latest observation
        if done:
            self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [], [], [], [], []

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # DONE :  fill in the call to the update function using the appropriate tensors
            log = self.q_fun.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n
            )
            
            # DONE : fill in the call to the update function using the appropriate tensors
            ## Hint the actor will need a copy of the q_net to maximize the Q-function
            if self.num_param_updates % self.policy_delay == 0:
                log.update(self.actor.update(
                    ob_no, self.q_fun.q_net
                ))

            # DONE : update the target network periodically
            if self.num_param_updates % self.target_update_freq == 0:
                self.q_fun.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log

    def save(self, path):
        pass
