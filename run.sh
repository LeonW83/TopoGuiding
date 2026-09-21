#!/bin/bash
set -e

# First command line argument is method to use in ["no_topoloss", "topoloss", "full_guiding", "stepwise_guiding"]
METHOD="$1"

SEEDS=(0 1 2 3 4)
MODELS=("Diffusion" "FlowMatching")
TEST_STEPS_LIST=(50 30)
NUM_GEN_SAMPLES=100

for MODEL in "${MODELS[@]}"; do
for TEST_STEPS in "${TEST_STEPS_LIST[@]}"; do
for SEED in  "${SEEDS[@]}"; do
      # NO GUIDING / NO TOPOLOSS
      if [[ "$METHOD" == "no_topoloss" ]]; then
          python main.py --model="${MODEL}" --dataset=shapes --data_dir=data --cache_dir=.cache --image_size=256 \
                  --train_steps=0 --eval_every_n_steps=625 --pos_encoding_dim=256 --batch_size=16 --topo_dim=0 --reg_weight=1e-5 \
                  --learning_rate=1e-4 --num_test_steps="${TEST_STEPS}" --num_gen_samples="${NUM_GEN_SAMPLES}" --device=cuda --seed="${SEED}" \
                  --results_dir="results/${MODEL}/steps_${TEST_STEPS}/no_topoloss" --results_name="seed_${SEED}" \
                  --model_checkpoint="./ckpts_${MODEL}_final/${MODEL}_wo_topoloss.ckpt" --guiding_epochs=0 --gen_batch_size=16

      # NO GUIDING / WITH TOPOLOSS
      elif [[ "$METHOD" == "topoloss" ]]; then
          python main.py --model="${MODEL}" --dataset=shapes --data_dir=data --cache_dir=.cache --image_size=256 \
                  --train_steps=0 --eval_every_n_steps=625 --pos_encoding_dim=256 --batch_size=16 --topo_dim=0 --reg_weight=1e-5 \
                  --learning_rate=1e-4 --num_test_steps="${TEST_STEPS}" --num_gen_samples="${NUM_GEN_SAMPLES}" --device=cuda --seed="${SEED}" \
                  --results_dir="results/${MODEL}/steps_${TEST_STEPS}/topoloss" --results_name="seed_${SEED}" \
                  --model_checkpoint="./ckpts_${MODEL}_final/${MODEL}_with_topoloss.ckpt" --guiding_epochs=0 --gen_batch_size=16

      # STEPWISE GUIDING
      elif [[ "$METHOD" == "full_guiding" ]]; then
          python main.py --model="${MODEL}" --dataset=shapes --data_dir=data --cache_dir=.cache --image_size=256 \
                  --train_steps=0 --eval_every_n_steps=625 --pos_encoding_dim=256 --batch_size=16 --topo_dim=0 --reg_weight=1e-5 \
                  --learning_rate=1e-4 --num_test_steps="${TEST_STEPS}" --num_gen_samples="${NUM_GEN_SAMPLES}" --device=cuda --seed="${SEED}" \
                  --results_dir="results/${MODEL}/steps_${TEST_STEPS}/full_guiding" --results_name="seed_${SEED}" \
                  --model_checkpoint="./ckpts_${MODEL}_final/${MODEL}_wo_topoloss.ckpt" --guiding_epochs=5 --gen_batch_size=1

      # FULL-PATH GUIDING
      elif [[ "$METHOD" == "stepwise_guiding" ]]; then
          python main.py --model="${MODEL}" --dataset=shapes --data_dir=data --cache_dir=.cache --image_size=256 \
                  --train_steps=0 --eval_every_n_steps=625 --pos_encoding_dim=256 --batch_size=16 --topo_dim=0 --reg_weight=1e-5 \
                  --learning_rate=1e-4 --num_test_steps="${TEST_STEPS}" --num_gen_samples="${NUM_GEN_SAMPLES}" --device=cuda --seed="${SEED}" \
                  --results_dir="results/${MODEL}/steps_${TEST_STEPS}/stepwise_guiding" --results_name="seed_${SEED}" \
                  --model_checkpoint="./ckpts_${MODEL}_final/${MODEL}_wo_topoloss.ckpt" --guiding_epochs=1 --stepwise_guiding --gen_batch_size=16

      # ABLATION ONLY
      elif [[ "$METHOD" == "random_ablation" ]]; then
          python main.py --model="${MODEL}" --dataset=shapes --data_dir=data --cache_dir=.cache --image_size=256 \
                  --train_steps=0 --eval_every_n_steps=625 --pos_encoding_dim=256 --batch_size=16 --topo_dim=0 --reg_weight=1e-5 \
                  --learning_rate=1e-4 --num_test_steps="${TEST_STEPS}" --num_gen_samples="${NUM_GEN_SAMPLES}" --device=cuda --seed="${SEED}" \
                  --results_dir="results/${MODEL}/steps_${TEST_STEPS}/stepwise_guiding" --results_name="seed_${SEED}" \
                  --model_checkpoint="./ckpts_${MODEL}_final/${MODEL}_wo_topoloss.ckpt" --guiding_epochs=5 --random_ablation --gen_batch_size=16
      fi

done; done; done

echo "All jobs done."