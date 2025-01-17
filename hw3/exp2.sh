#!/bin/bash

BS=5000
LR=0.001
python run_hw3.py env_name=InvertedPendulum-v2 \
                  ep_len=1000 \
                  estimate_advantage_args.discount=0.9 \
                  n_iter=100\
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=64 \
                  batch_size=${BS} \
                  train_batch_size=${BS} \
                  eval_batch_size=${BS} \
                  computation_graph_args.learning_rate=${LR}\
                  estimate_advantage_args.reward_to_go=True \
                  exp_name=q2_b${BS}_lr${LR} \
                  rl_alg=reinforce\
                  no_gpu=true

LR=0.01
python run_hw3.py env_name=InvertedPendulum-v2 \
                  ep_len=1000 \
                  estimate_advantage_args.discount=0.9 \
                  n_iter=100\
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=64 \
                  batch_size=${BS} \
                  train_batch_size=${BS} \
                  eval_batch_size=${BS} \
                  computation_graph_args.learning_rate=${LR}\
                  estimate_advantage_args.reward_to_go=True \
                  exp_name=q2_b${BS}_lr${LR} \
                  rl_alg=reinforce\
                  no_gpu=true

LR=0.1
python run_hw3.py env_name=InvertedPendulum-v2 \
                  ep_len=1000 \
                  estimate_advantage_args.discount=0.9 \
                  n_iter=100\
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=64 \
                  batch_size=${BS} \
                  train_batch_size=${BS} \
                  eval_batch_size=${BS} \
                  computation_graph_args.learning_rate=${LR}\
                  estimate_advantage_args.reward_to_go=True \
                  exp_name=q2_b${BS}_lr${LR} \
                  rl_alg=reinforce\
                  no_gpu=true
