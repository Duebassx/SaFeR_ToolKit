#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=4,5,6,7


MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

TRAIN_DATA=""
VAL_DATA=""
IMAGE_DIR=null
N_GPUS=4

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.image_dir=${IMAGE_DIR} \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.rollout_batch_size=256 \
    data.val_batch_size=128 \
    data.format_prompt=./examples/format_prompt/toolkit.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=32 \
    worker.actor.optim.lr=1.0e-6 \
    worker.rollout.n=4 \
    worker.rollout.temperature=1.0 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.reward.reward_function=./examples/reward_function/safer_toolkit.py:compute_score \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.total_epochs=2 \
    trainer.val_freq=50 \
    trainer.save_freq=50 \
    trainer.logger='["file","wandb"]'

