#!/bin/bash

python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  batch_size=1000 \
                  exp_name=q6_1_1 \
                  computation_graph_args.num_target_updates=1 \
                  computation_graph_args.num_grad_steps_per_target_update=1 \
                  rl_alg=ac\
                  no_gpu=False

python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  rl_alg=ac \
                  batch_size=1000 \
                  eval_batch_size=1000 \
                  exp_name=q6_100_1 \
                  computation_graph_args.num_target_updates=100 \
                  computation_graph_args.num_grad_steps_per_target_update=1\
                  no_gpu=False


python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  rl_alg=ac \
                  batch_size=1000 \
                  eval_batch_size=1000 \
                  exp_name=q6_1_100 \
                  computation_graph_args.num_target_updates=1 \
                  computation_graph_args.num_grad_steps_per_target_update=100\
                  no_gpu=False

python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  rl_alg=ac \
                  batch_size=1000 \
                  eval_batch_size=1000 \
                  exp_name=q6_10_10 \
                  computation_graph_args.num_target_updates=10 \
                  computation_graph_args.num_grad_steps_per_target_update=10\
                  no_gpu=False