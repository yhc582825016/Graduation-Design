#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT
export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
deepspeed --include localhost:0,1,2,3,4,5,7 main.py --model_name_or_path /workspace/ye/llm/opt-1b3 \
                     --data_path /workspace/ye/dataset/processed_Dahoas/full_hh_rlhf \
                    --gradient_accumulation_steps 1 --zero_stage $ZERO_STAGE \
                    --per_device_train_batch_size 4 \
                    --per_device_eval_batch_size 4 \
                    --enable_wandb \
                    --project_name RLHF-PPO \
                    --run_name opt-1.3b-sft-hh-rlhf-only \
                    --max_seq_len 1024 \
                    --num_train_epochs 2 \
                    --eval_ratio 0.2 \
                    --deepspeed --output_dir $OUTPUT 
# &> $OUTPUT/training.log