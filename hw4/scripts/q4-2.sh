##!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=50000
LR=0.002

TESTED_UPDATE_FREQ=(1, 2, 4)
for UPDATE_FREQ in "${TESTED_UPDATE_FREQ[@]}"
do
  for ((i=0;i<3;i+=1))
  do
    python run_hw4.py exp_name=q4_ddpg_up${UPDATE_FREQ}_seed${i} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  policy_delay=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${i}
  done
done
