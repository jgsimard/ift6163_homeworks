##!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=200000
LR=0.002
UPDATE_FREQ=2
python run_hw4.py exp_name=q5_ddpg \
                    rl_alg=ddpg \
                    env_name=HalfCheetah-v2 \
                    atari=false \
                    no_gpu=true \
                    discrete=false \
                    n_iter=${N_ITER} \
                    critic_learning_rate=${LR} \
                    learning_rate=${LR} \
                    policy_delay=${UPDATE_FREQ}
