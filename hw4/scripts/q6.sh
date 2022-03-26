export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=50000
LR=0.001

TESTED_RHO=(0.1, 0.2, 0.5)
for RHO in "${TESTED_RHO[@]}"
do
  for ((i=0;i<3;i+=1))
  do
    python run_hw4.py exp_name=q6_td3_rho${RHO}_seed${i} rl_alg=td3 env_name=InvertedPendulum-v2 atari=false no_gpu=true discrete=false n_iter=${N_ITER} td3_target_policy_noise=${RHO} learning_rate=${LR} critic_learning_rate=${LR} seed=${i}
  done
done