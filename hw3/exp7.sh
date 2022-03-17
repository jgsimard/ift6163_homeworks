NTU=10
NGSPTU=10

python run_hw3.py env_name=InvertedPendulum-v2 \
                  rl_alg=ac \
                  ep_len=1000 \
                  estimate_advantage_args.discount=0.95 \
                  n_iter=100 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=64 \
                  batch_size=5000 \
                  train_batch_size=5000 \
                  computation_graph_args.learning_rate=0.01 \
                  exp_name=q7_InvertedPendulum_${NTU}_${NGSPTU} \
                  computation_graph_args.num_target_updates=${NTU} \
                  computation_graph_args.num_grad_steps_per_target_update=${NGSPTU}\
                  no_gpu=true

python run_hw3.py env_name=HalfCheetah-v2 \
                  rl_alg=ac \
                  ep_len=150 \
                  estimate_advantage_args.discount=0.90\
                  scalar_log_freq=5 \
                  n_iter=150 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=32 \
                  batch_size=30000 \
                  train_batch_size=30000 \
                  eval_batch_size=1500 \
                  computation_graph_args.learning_rate=0.02 \
                  exp_name=q7_HalfCheetah_${NTU}_${NGSPTU} \
                  computation_graph_args.num_target_updates=${NTU} \
                  computation_graph_args.num_grad_steps_per_target_update=${NGSPTU}\
                  no_gpu=true