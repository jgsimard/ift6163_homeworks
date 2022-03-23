#!/bin/bash

LR=0.001
python run_hw4.py exp_name=q4_ddpg_up_lr${LR} \
                  rl_alg=td3 \
                  env_name=InvertedPendulum-v2 \
                  atari=false \
                  no_gpu=true \
                  learning_freq=1 \
                  discrete=false \
                  learning_starts=64 \
                  n_iter=10000 \
                  video_log_freq=100

# in TD3 paper, they recommend