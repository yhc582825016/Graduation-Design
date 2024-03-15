#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT
RANDOM_PORT=$RANDOM
echo $RANDOM_PORT
export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=$RANDOM_PORT main.py --model_name_or_path /workspace/ye/llm/opt-1b3 \
            --data_path  /workspace/ye/dataset/processed_Dahoas/full_hh_rlhf\
            --num_padding_at_beginning 1 --weight_decay 0.1 --dropout 0.0 --gradient_accumulation_steps 1 --zero_stage $ZERO_STAGE \
            --enable_wandb \
            --max_seq_len 1024 \
            --per_device_train_batch_size 2 \
            --project_name RLHF-PPO \
            --run_name RM-hh-dataset-130-processed-data \
            --deepspeed --output_dir $OUTPUT 
#&> $OUTPUT/RM-hh-dataset-201-processed-data &