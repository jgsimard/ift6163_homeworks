#!/bin/bash
# batch_size=1000
python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  batch_size=1000 \
                  train_batch_size=1000 \
                  estimate_advantage_args.standardize_advantages=False \
                  exp_name=q1_sb_no_rtg_dsa \
                  rl_alg=reinforce

python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  batch_size=1000 \
                  train_batch_size=1000 \
                  estimate_advantage_args.reward_to_go=True \
                  estimate_advantage_args.standardize_advantages=False \
                  exp_name=q1_sb_rtg_dsa \
                  rl_alg=reinforce

python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  batch_size=1000 \
                  train_batch_size=1000 \
                  estimate_advantage_args.reward_to_go=True \
                  exp_name=q1_sb_rtg_na \
                  rl_alg=reinforce


# batch_size=5000
python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  batch_size=5000 \
                  train_batch_size=5000 \
                  estimate_advantage_args.standardize_advantages=False \
                  exp_name=q1_lb_no_rtg_dsa \
                  rl_alg=reinforce

python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  batch_size=5000 \
                  train_batch_size=5000 \
                  estimate_advantage_args.reward_to_go=True \
                  estimate_advantage_args.standardize_advantages=False \
                  exp_name=q1_lb_rtg_dsa \
                  rl_alg=reinforce

python run_hw3.py env_name=CartPole-v0 \
                  n_iter=100 \
                  batch_size=5000 \
                  train_batch_size=5000 \
                  estimate_advantage_args.reward_to_go=True \
                  exp_name=q1_lb_rtg_na \
                  rl_alg=reinforce