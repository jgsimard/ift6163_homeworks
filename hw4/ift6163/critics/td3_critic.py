from .ddpg_critic import DDPGCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy

from ift6163.infrastructure import pytorch_util as ptu
from ift6163.policies.MLP_policy import ConcatMLP
from ift6163.critics.ddpg_critic import polyak_update


class TD3Critic(DDPGCritic):
    def __init__(self, actor, hparams, optimizer_spec, env, **kwargs):
        super().__init__(actor, hparams, optimizer_spec, **kwargs)
        self.target_policy_noise = hparams['td3_target_policy_noise']
        self.noise_clip = hparams['td3_target_policy_noise_clip']
        self.action_min = ptu.from_numpy(env.action_space.low)
        self.action_max = ptu.from_numpy(env.action_space.high)

        # create the Twin critic network and the corresponding target
        hparams = copy.deepcopy(hparams)
        hparams['ob_dim'] = hparams['ob_dim'] + hparams['ac_dim']
        self.q_net_2 = ConcatMLP(
            hparams['ac_dim'],
            hparams['ob_dim'],
            hparams['n_layers_critic'],
            hparams['size_hidden_critic'],
            discrete=False,
            learning_rate=hparams['critic_learning_rate'],
            nn_baseline=False,
            deterministic=True,
            activation=hparams['activation']
        )
        self.q_net_target_2 = ConcatMLP(
            hparams['ac_dim'],
            hparams['ob_dim'],
            hparams['n_layers_critic'],
            hparams['size_hidden_critic'],
            discrete=False,
            learning_rate=hparams['critic_learning_rate'],
            nn_baseline=False,
            deterministic=True,
            activation=hparams['activation']
        )

        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.q_net_2.parameters()),
            self.learning_rate,
        )
        self.q_net_2.to(ptu.device)
        self.q_net_target_2.to(ptu.device)

    def update_target_network(self):
        super(TD3Critic, self).update_target_network()
        polyak_update(self.q_net_target_2, self.q_net_2, self.polyak_avg)

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
        # numpy -> pytorch
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # current q value estimates
        q_t_values_1 = self.q_net(ob_no, ac_na).squeeze()
        q_t_values_2 = self.q_net_2(ob_no, ac_na).squeeze()

        # DONE : compute the Q-values from the target network
        with torch.no_grad():
            # compute clipped action
            max_action_tp1 = self.actor_target(next_ob_no)
            action_noise = torch.randn(self.ac_dim) * self.target_policy_noise # sample action noise
            action_noise = action_noise.clamp(-self.noise_clip, self.noise_clip) # clip action noise
            action = max_action_tp1 + action_noise.to(ptu.device) # add action noise to action
            action = action.clamp(self.action_min, self.action_max) # clip noisy action

            # next q_value estimates
            q_tp1_values_1 = self.q_net_target(next_ob_no, action).squeeze()
            q_tp1_values_2 = self.q_net_target_2(next_ob_no, action).squeeze()

            # take the minimum next q_value
            q_tp1_values = torch.min(q_tp1_values_1, q_tp1_values_2)

            # DONE : compute targets for minimizing Bellman error
            if len(q_t_values_1.shape) == 2:  # this is horrible code but it works for now
                reward_n = reward_n.view(-1, 1)
                terminal_n = terminal_n.view(-1, 1)
            target = reward_n + self.gamma + q_tp1_values * (1 - terminal_n)
            # target = target.detach()

        # compute the loss
        assert q_t_values_1.shape == target.shape, f"q_t_values_1={q_t_values_1.shape}, target={target.shape}"
        assert q_t_values_2.shape == target.shape, f"q_t_values_2={q_t_values_2.shape}, target={target.shape}"
        loss = self.loss(q_t_values_1, target) + self.loss(q_t_values_2, target)

        # optimize the critic
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(list(self.q_net.parameters()) + list(self.q_net_2.parameters()), self.grad_norm_clipping)
        self.optimizer.step()
        # self.learning_rate_scheduler.step()
        return {
            'Training_Loss': ptu.to_numpy(loss),
        }


