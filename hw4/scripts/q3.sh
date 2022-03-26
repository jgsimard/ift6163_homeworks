TESTED_LR=(0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01)
for LR in "${TESTED_LR[@]}"
do
  for ((i=0;i<3;i+=1))
  do
    python run_hw4.py env_name=LunarLander-v3 exp_name=q3_lr${LR}_seed${i} double_q=true seed=${i} n_iter=250000 learning_rate=${LR} no_gpu=True
  done
done
