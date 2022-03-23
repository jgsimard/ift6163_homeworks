#!/bin/bash

#python run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_1 seed=1 n_iter=250000
#python run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_2 seed=2 n_iter=250000
#python run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_3 seed=3 n_iter=250000

#python run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_1 double_q=true seed=1 n_iter=250000
#python run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_2 double_q=true seed=2 n_iter=250000
#python run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_3 double_q=true seed=3 n_iter=250000

LR=0.001
python run_hw4.py exp_name=q4_ddpg_up_lr${LR} \
                  rl_alg=ddpg \
                  env_name=InvertedPendulum-v2 \
                  atari=false \
                  no_gpu=true \
                  learning_freq=2 \
                  discrete=false \
                  learning_starts=1024 \
                  n_iter=10000 \
                  video_log_freq=100


#python run_hw4.py exp_name=q4_ddpg_up<b>_lr${LR} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false
#python run_hw4.py exp_name=q4_ddpg_up<b>_lr${LR} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false