##!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=50000
LR=0.002

UPDATE_FREQ=1
SEED=1
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}

UPDATE_FREQ=2
SEED=1
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}


UPDATE_FREQ=4
SEED=1
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${SEED} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
