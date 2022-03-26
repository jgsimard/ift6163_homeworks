export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=200000
#LR=0.0003
LR=0.002
RHO=0.2
SIZE=256
python run_hw4.py exp_name=q7_td3\
                   rl_alg=td3 \
                   env_name=HalfCheetah-v2 \
                   atari=false \
                   discrete=false \
                   n_iter=${N_ITER} \
                   td3_target_policy_noise=${RHO} \
                   critic_learning_rate=${LR} \
                   learning_rate=${LR} \
                   size_hidden_critic=${SIZE} \
                   size=${SIZE}\
                   batch_size=256\
                   train_batch_size=256\
                   learning_starts=20000 \
                   activation=relu\
                   polyak_avg=0.005



