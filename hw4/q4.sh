##!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=50000
UPDATE_FREQ=1

LR=0.0002
SEED=1
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}

LR=0.0005
#SEED=1
#python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}

LR=0.001
SEED=1
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}

LR=0.002
SEED=1
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
