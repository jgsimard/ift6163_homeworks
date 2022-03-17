# EXP 8
#BS=5000
#GEN=false
#python run_hw3.py exp_name=q8_b${BS}_gen${GEN} \
#                  env_name=cheetah-ift6163-v0 estimate_advantage_args.discount=0.95 \
#                  computation_graph_args.n_layers=2 \
#                  computation_graph_args.size=32 \
#                  computation_graph_args.learning_rate=0.01 \
#                  n_iter=100 \
#                  batch_size=${BS} \
#                  train_batch_size=1024 \
#                  eval_batch_size=2000 \
#                  rl_alg=dyna \
#                  no_gpu=true \
#                  train_args.use_gen_data=${GEN}
#
#GEN=true
#python run_hw3.py exp_name=q8_b${BS}_gen${GEN} \
#                  env_name=cheetah-ift6163-v0 estimate_advantage_args.discount=0.95 \
#                  computation_graph_args.n_layers=2 \
#                  computation_graph_args.size=32 \
#                  computation_graph_args.learning_rate=0.01 \
#                  n_iter=100 \
#                  batch_size=${BS} \
#                  train_batch_size=1024 \
#                  eval_batch_size=2000 \
#                  rl_alg=dyna \
#                  no_gpu=true \
#                  train_args.use_gen_data=${GEN}
BS=2000
GEN=false
python run_hw3.py exp_name=q8_b${BS}_gen${GEN} \
                  env_name=cheetah-ift6163-v0 estimate_advantage_args.discount=0.95 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=32 \
                  computation_graph_args.learning_rate=0.01 \
                  n_iter=100 \
                  batch_size=${BS} \
                  train_batch_size=1024 \
                  eval_batch_size=2000 \
                  rl_alg=dyna \
                  no_gpu=true \
                  train_args.use_gen_data=${GEN}
GEN=true
python run_hw3.py exp_name=q8_b${BS}_gen${GEN} \
                  env_name=cheetah-ift6163-v0 estimate_advantage_args.discount=0.95 \
                  computation_graph_args.n_layers=2 \
                  computation_graph_args.size=32 \
                  computation_graph_args.learning_rate=0.01 \
                  n_iter=100 \
                  batch_size=${BS} \
                  train_batch_size=1024 \
                  eval_batch_size=2000 \
                  rl_alg=dyna \
                  no_gpu=true \
                  train_args.use_gen_data=${GEN}
