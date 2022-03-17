#!/bin/bash

python run_hw3.py env_name=LunarLanderContinuous-v2 \
                  ep_len=1000 \
                  estimate_advantage_args.discount=0.99 \
                  n_iter=100 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=64 \
                  batch_size=40000 \
                  train_batch_size=40000 \
                  eval_batch_size=4000 \
                  computation_graph_args.learning_rate=0.005 \
                  estimate_advantage_args.reward_to_go=true \
                  estimate_advantage_args.nn_baseline=true \
                  rl_alg=reinforce \
                  exp_name=q3_b40000_r0.005 \
                  no_gpu=true