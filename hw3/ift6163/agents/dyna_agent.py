from collections import OrderedDict

from .base_agent import BaseAgent
from ift6163.models.ff_model import FFModel
from ift6163.policies.MLP_policy import MLPPolicyPG
from ift6163.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.infrastructure.utils import *
from ift6163.infrastructure import pytorch_util as ptu


class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        for i in range(self.ensemble_size):

            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful

            # from hw2
            index_low = i * num_data_per_ens
            index_high = (i + 1) * num_data_per_ens
            observations = ob_no[index_low:index_high]  # DONE(Q1)
            actions = ac_na[index_low:index_high]  # DONE(Q1)
            next_observations = next_ob_no[index_low:index_high]  # DONE(Q1)
            model = self.dyn_models[i]  # DONE(Q1)

            # Copy this from previous homework
            log = model.update(observations, actions, next_observations, self.data_statistics)
            loss = log['Training_Loss']
            losses.append(loss)
            
        # DONE :  Pick a model at random
        model = self.dyn_models[np.random.choice(len(self.dyn_models))]
        # DONE Use that model to generate one additional next_ob_no for every state in ob_no (using the policy distribution)
        # Hint: You may need the env to label the rewards
        # Hint: Keep things on policy
        generated_actions = self.actor.get_action(ob_no)
        generated_obs = model.get_prediction(ob_no, ac_na, self.data_statistics)
        generated_rewards, generated_terminals = self.env.get_reward(ob_no, generated_actions)
        
        # DONE : add this generated data to the real data
        ob_no_plus_gen = np.concatenate((ob_no, ob_no))
        ac_na_plus_gen = np.concatenate((ac_na, generated_actions))
        next_ob_no_plus_gen = np.concatenate((next_ob_no, generated_obs))
        re_n_plus_gen = np.concatenate((re_n, generated_rewards))
        terminal_n_plus_gen = np.concatenate((terminal_n, generated_terminals))

        # DONE :  Perform a policy gradient update (copy from ac_agent)
        # Hint: Should the critic be trained with this generated data?
        # Try with and without and include your findings in the report. =>s ok.
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            if self.agent_params['use_gen_data']:
                critic_loss = self.critic.update(ob_no_plus_gen, ac_na_plus_gen, next_ob_no_plus_gen, re_n_plus_gen, terminal_n_plus_gen)
            else:
                critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        advantage = self.estimate_advantage(ob_no_plus_gen, next_ob_no_plus_gen, re_n_plus_gen, terminal_n_plus_gen)
        #     update the actor
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no_plus_gen, ac_na_plus_gen, advantage)

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss['Training_Loss']
        loss['FD_Loss'] = np.mean(losses)
        return loss

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            'delta_std': np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        # return self.replay_buffer.sample_random_data(batch_size * self.ensemble_size)
        return self.replay_buffer.sample_recent_data(batch_size * self.ensemble_size)

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # DONE :  Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)

        # numpy -> pytorch
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n).bool()

        # 1) query the critic with ob_no, to get V(s)
        v_s = self.critic(ob_no)
        # 2) query the critic with next_ob_no, to get V(s')
        v_s_prime = self.critic(next_ob_no)
        # cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        v_s_prime[terminal_n] = 0
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        q_s_a = re_n + self.critic.gamma * v_s_prime
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        adv_n = q_s_a - v_s
        # pytorch -> numpy
        adv_n = ptu.to_numpy(adv_n)

        if self.agent_params.standardize_advantages:
            adv_n = normalize(adv_n, np.mean(adv_n), np.std(adv_n))
        return adv_n

    def save(self, path):
        pass
