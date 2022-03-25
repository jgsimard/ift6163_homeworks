export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=50000
LR=0.001
RHO=0.1
SEED=1
python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${SEED} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${SEED} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${SEED} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}

RHO=0.2
SEED=1
python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${SEED} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${SEED} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${SEED} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}

RHO=0.5
SEED=1
python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${SEED} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=2
python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${SEED} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}
SEED=3
python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${SEED} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${SEED}


