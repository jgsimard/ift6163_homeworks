from .ddpg_critic import DDPGCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy

from ift6163.infrastructure import pytorch_util as ptu
from ift6163.policies.MLP_policy import ConcatMLP


class TD3Critic(DDPGCritic):
    def __init__(self, actor, hparams, optimizer_spec, env, **kwargs):
        hparams['twin'] = True  # for the Twin part of Twin Delayed DDPG (TD3)
        super().__init__(actor, hparams, optimizer_spec, **kwargs)
        self.target_policy_noise = hparams['td3_target_policy_noise']
        self.noise_clip = hparams['td3_target_policy_noise_clip']
        self.action_min = ptu.from_numpy(env.action_space.low)
        self.action_max = ptu.from_numpy(env.action_space.high)
        self.twin = hparams['twin']

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        # max_ac = ptu.from_numpy(max_action)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        q_t_values = self.q_net(ob_no, ac_na).squeeze()
        if self.twin:
            q_t_values = q_t_values.view(-1, 2)

        # DONE : compute the Q-values from the target network
        ## Hint: you will need to use the target policy

        # print(f"self.ac_dim={self.ac_dim}")
        # compute clipped action
        max_action_tp1 = self.actor_target(next_ob_no)
        action_noise = torch.randn(self.ac_dim) * self.target_policy_noise
        action_noise_clip = torch.clamp(action_noise, min=-self.noise_clip, max=self.noise_clip)
        action = max_action_tp1 + action_noise_clip
        action_clip = torch.clamp(action, min=self.action_min, max=self.action_max)


        q_tp1_values = self.q_net_target(next_ob_no, action_clip).squeeze()
        if self.twin:
            q_tp1_values = q_tp1_values.view(-1, 2)
            # choose the min value of the two
            q_tp1_values, _ = q_tp1_values.min(dim=1)
            # print(q_tp1_values)
            # print(q_tp1_values.shape)

        # DONE : compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        # print(f"terminal_n={terminal_n.shape}")
        # reward_n = reward_n.view(-1, 1)
        #
        # terminal_n = terminal_n.view(-1, 1)
        target = reward_n + self.gamma + q_tp1_values * (1 - terminal_n)
        target = target.detach()
        if self.twin:
            target = target.view(-1, 1)
            target = target.expand(-1, 2)

        assert q_t_values.shape == target.shape, f"q_t_values={q_t_values.shape}, target={target.shape}"
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        # self.learning_rate_scheduler.step()
        return {
            'Training_Loss': ptu.to_numpy(loss),
        }


