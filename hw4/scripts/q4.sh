##!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
N_ITER=50000
UPDATE_FREQ=1

TESTED_LR=(0.0002, 0.0005, 0.001, 0.002)
for LR in "${TESTED_LR[@]}"
do
  for ((i=0;i<3;i+=1))
  do
    python run_hw4.py exp_name=q4_ddpg_lr${LR}_seed${i} rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false no_gpu=true  learning_freq=${UPDATE_FREQ}  discrete=false n_iter=${N_ITER} learning_rate=${LR} critic_learning_rate=${LR} seed=${i}
  done
done
