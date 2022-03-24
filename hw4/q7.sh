export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=100000
#RHO=0.1
#python run_hw4.py exp_name=q6_td3_rho${RHO} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=0.0005
#RHO=0.2
#python run_hw4.py exp_name=q6_td3_rho${RHO} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=0.0005
#RHO=0.5
#python run_hw4.py exp_name=q6_td3_rho${RHO} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=0.0005


RHO=0.1
UPDATE_FREQ=1
python run_hw4.py exp_name=q6_td3_rho${RHO}_up${UPDATE_FREQ} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=0.0005 policy_delay=${UPDATE_FREQ}

#RHO=0.1
#UPDATE_FREQ=4
#python run_hw4.py exp_name=q6_td3_rho${RHO}_up${UPDATE_FREQ} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=0.0005 policy_delay=${UPDATE_FREQ}
