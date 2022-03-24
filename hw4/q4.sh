##!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=100000
#UPDATE_FREQ=1
#LR=0.001
#python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_lr${LR} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER}
#LR=0.005
#python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_lr${LR} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER}
#LR=0.01
#python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_lr${LR} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER}

LR=0.001
UPDATE_FREQ=2
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_lr${LR} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER}
UPDATE_FREQ=4
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_lr${LR} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER}
