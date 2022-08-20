import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            obs = obs
        else:
            obs = obs[None]
        
        ## DONE : return the action that maxinmizes the Q-value
        # at the current observation as the output
        qa_values = self.critic.qa_values(obs)
        # print(f"obs.shape={obs.shape}, qa_values={qa_values.shape}")
        action = qa_values.argmax(1)[0]
        # action = action.squeeze()
        # print(f"action.shape={action.shape}, action={action}, type={type(action)}")
        return action