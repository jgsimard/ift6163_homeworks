export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=100000
RHO=0.1
UPDATE_FREQ=2
python run_hw4.py exp_name=test_td3_rho${RHO}_up${UPDATE_FREQ} \
                    rl_alg=td3 \
                    env_name=InvertedPendulum-v2 \
                    atari=false \
                    no_gpu=true \
                    discrete=false \
                    n_iter=${N_ITER} \
                    td3_target_policy_noise=${RHO} \
                    learning_rate=0.0005 \
                    policy_delay=${UPDATE_FREQ}\
                    batch_size=256 \
                    train_batch_size=256

#LR=0.0005
#python run_hw4.py exp_name=test_ddpg_up${UPDATE_FREQ}_lr${LR} \
#                    rl_alg=ddpg \
#                    env_name=InvertedPendulum-v2 \
#                    atari=false \
#                    no_gpu=true  \
#                    discrete=false \
#                    n_iter=${N_ITER}


