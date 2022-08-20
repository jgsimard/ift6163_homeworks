#!/bin/bash

python run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_1 seed=1 n_iter=250000
python run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_2 seed=2 n_iter=250000
python run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_3 seed=3 n_iter=250000

python run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_1 double_q=true seed=1 n_iter=250000
python run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_2 double_q=true seed=2 n_iter=250000
python run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_3 double_q=true seed=3 n_iter=250000