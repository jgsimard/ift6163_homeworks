export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=200000
LR=0.001
RHO=0.2
SIZE=64
python run_hw4.py exp_name=q7_td3\
                   rl_alg=td3 \
                   env_name=HalfCheetah-v2 \
                   atari=false \
                   no_gpu=true \
                   discrete=false \
                   n_iter=${N_ITER} \
                   td3_target_policy_noise=${RHO} \
                   critic_learning_rate=${LR} \
                   learning_rate=${LR} \
                   size_hidden_critic=${SIZE} \
                   size=${SIZE}
