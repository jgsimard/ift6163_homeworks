#!/bin/bash

# batch size 10000
BS=50000
LR=0.005
python run_hw3.py env_name=HalfCheetah-v2 \
                  ep_len=150 \
                  estimate_advantage_args.discount=0.95 \
                  n_iter=100 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=32 \
                  batch_size=${BS} \
                  train_batch_size=${BS} \
                  eval_batch_size=1000 \
                  computation_graph_args.learning_rate=${LR} \
                  estimate_advantage_args.reward_to_go=true \
                  estimate_advantage_args.nn_baseline=true \
                  rl_alg=reinforce \
                  exp_name=q4_search_b${BS}_lr${LR}_rtg_nnbaseline \
                  no_gpu=true
LR=0.01
python run_hw3.py env_name=HalfCheetah-v2 \
                  ep_len=150 \
                  estimate_advantage_args.discount=0.95 \
                  n_iter=100 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=32 \
                  batch_size=${BS} \
                  train_batch_size=${BS} \
                  eval_batch_size=1000 \
                  computation_graph_args.learning_rate=${LR} \
                  estimate_advantage_args.reward_to_go=true \
                  estimate_advantage_args.nn_baseline=true \
                  rl_alg=reinforce \
                  exp_name=q4_search_b${BS}_lr${LR}_rtg_nnbaseline \
                  no_gpu=true
LR=0.02
python run_hw3.py env_name=HalfCheetah-v2 \
                  ep_len=150 \
                  estimate_advantage_args.discount=0.95 \
                  n_iter=100 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=32 \
                  batch_size=${BS} \
                  train_batch_size=${BS} \
                  eval_batch_size=1000 \
                  computation_graph_args.learning_rate=${LR} \
                  estimate_advantage_args.reward_to_go=true \
                  estimate_advantage_args.nn_baseline=true \
                  rl_alg=reinforce \
                  exp_name=q4_search_b${BS}_lr${LR}_rtg_nnbaseline \
                  no_gpu=true

BS=30000
LR=0.005
python run_hw3.py env_name=HalfCheetah-v2 \
                  ep_len=150 \
                  estimate_advantage_args.discount=0.95 \
                  n_iter=100 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=32 \
                  batch_size=${BS} \
                  train_batch_size=${BS} \
                  eval_batch_size=1000 \
                  computation_graph_args.learning_rate=${LR} \
                  estimate_advantage_args.reward_to_go=true \
                  estimate_advantage_args.nn_baseline=true \
                  rl_alg=reinforce \
                  exp_name=q4_search_b${BS}_lr${LR}_rtg_nnbaseline \
                  no_gpu=true
LR=0.01
python run_hw3.py env_name=HalfCheetah-v2 \
                  ep_len=150 \
                  estimate_advantage_args.discount=0.95 \
                  n_iter=100 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=32 \
                  batch_size=${BS} \
                  train_batch_size=${BS} \
                  eval_batch_size=1000 \
                  computation_graph_args.learning_rate=${LR} \
                  estimate_advantage_args.reward_to_go=true \
                  estimate_advantage_args.nn_baseline=true \
                  rl_alg=reinforce \
                  exp_name=q4_search_b${BS}_lr${LR}_rtg_nnbaseline \
                  no_gpu=true
LR=0.02
python run_hw3.py env_name=HalfCheetah-v2 \
                  ep_len=150 \
                  estimate_advantage_args.discount=0.95 \
                  n_iter=100 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=32 \
                  batch_size=${BS} \
                  train_batch_size=${BS} \
                  eval_batch_size=1000 \
                  computation_graph_args.learning_rate=${LR} \
                  estimate_advantage_args.reward_to_go=true \
                  estimate_advantage_args.nn_baseline=true \
                  rl_alg=reinforce \
                  exp_name=q4_search_b${BS}_lr${LR}_rtg_nnbaseline \
                  no_gpu=true