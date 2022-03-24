from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy

from ift6163.infrastructure import pytorch_util as ptu
from ift6163.policies.MLP_policy import ConcatMLP


def polyak(target_param, param, weight):
    target_param.data.copy_(param.data * weight + target_param.data * (1.0 - weight))

class DDPGCritic(BaseCritic):

    def __init__(self, actor, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']
        self.learning_rate = hparams['critic_learning_rate']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']
        if hparams['twin']:
            out_size = 2
        else:
            out_size = 1


        self.optimizer_spec = optimizer_spec
        hparams = copy.deepcopy(hparams)
        hparams['ob_dim'] = hparams['ob_dim'] + hparams['ac_dim']
        print(f"hparams['ac_dim']={hparams['ac_dim']}")
        # hparams['ac_dim'] = 1
        self.q_net = ConcatMLP(   
            hparams['ac_dim'] * out_size,
            hparams['ob_dim'],
            hparams['n_layers_critic'],
            hparams['size_hidden_critic'],
            discrete=False,
            learning_rate=hparams['critic_learning_rate'],
            nn_baseline=False,
            deterministic=True
            )
        self.q_net_target = ConcatMLP(   
            hparams['ac_dim'] * out_size,
            hparams['ob_dim'],
            hparams['n_layers_critic'],
            hparams['size_hidden_critic'],
            discrete=False,
            learning_rate=hparams['critic_learning_rate'],
            nn_baseline=False,
            deterministic=True
            )
        # self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     self.optimizer_spec.learning_rate_schedule,
        # 
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            self.learning_rate,
            )
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)

        self.polyak_avg = hparams['polyak_avg']

        print("Critic")
        print(self.q_net)

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

        ### Hint: 
        # qa_t_values = self.q_net(ob_no, ac_na)
        # print(f"ob_no={ob_no.shape}, ac_na={ac_na.shape}")
        ac_na = ac_na.view(-1, 1)
        # print(f"ob_no={ob_no.shape}, ac_na={ac_na.shape}")
        q_t_values = self.q_net(ob_no, ac_na).squeeze() # the only thing that will be updated in this update function
        
        # DONE (maybe) compute the Q-values from the target network
        ## Hint: you will need to use the target policy
        max_action_tp1 = self.actor_target(next_ob_no)
        q_tp1_values = self.q_net_target(next_ob_no, max_action_tp1).squeeze()
        # print(f"max_action={max_action.shape}, q_tp1_values={q_tp1_values.shape}, next_ob_no={next_ob_no.shape}")

        # DONE :  compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        # print(reward_n.shape, terminal_n.shape)
        # reward_n = reward_n.view(-1, 1)
        target = reward_n + self.gamma + q_tp1_values * (1 - terminal_n)
        target = target.detach()

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

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            # Perform Polyak averaging
            polyak(target_param, param, self.polyak_avg)
        for target_param, param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            # Perform Polyak averaging for the target policy
            polyak(target_param, param, self.polyak_avg)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        ## HINT: the q function take two arguments  
        qa_values = TODO
        return ptu.to_numpy(qa_values)
