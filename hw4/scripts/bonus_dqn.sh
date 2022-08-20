LR=0.0002
#for ((i=0;i<3;i+=1))
#  do
#    python run_hw4.py env_name=LunarLander-v3 exp_name=bonus_dqn_double_dueling_lr${LR}_seed${i} double_q=true seed=${i} n_iter=250000 learning_rate=${LR} no_gpu=True dueling=True
#  done

for ((i=0;i<3;i+=1))
  do
    python run_hw4.py env_name=LunarLander-v3 exp_name=bonus_dqn_double_dueling_noisy_lr${LR}_seed${i} double_q=true seed=${i} n_iter=250000 learning_rate=${LR} no_gpu=True dueling=True noisy_net=True
  done
