#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# pip install deepspeed
# pip install transformers
# pip install sentencepiece
# pip install datasets
# pip install accelerate
# pip install wandb
# pip install loguru
# pip install yacs
# pip install nvitop
# pip install jsonlines
# pip install peft
# pip install trl
# pip install bitsandbytes

export PYTHONPATH=$PYTHONPATH:/workspace/ye/DeepSpeed-Chat
export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
# export WANDB_PROJECT=RLHF-PPO
# export WANDB_RUN_ID='ppo_opt_127'
# DeepSpeed Team
ACTOR_MODEL_PATH=/workspace/ye/DeepSpeed-Chat/training/step1_supervised_finetuning/output
CRITIC_MODEL_PATH=/workspace/ye/DeepSpeed-Chat/training/step2_reward_model_finetuning/output
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=/workspace/ye/DeepSpeed-Chat/training/step3_rlhf_finetuning/output
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT
RANDOM_PORT=$RANDOM
echo $RANDOM_PORT
deepspeed --include localhost:6 main.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --actor_zero_stage $ACTOR_ZERO_STAGE --critic_zero_stage $CRITIC_ZERO_STAGE \
   --data_path /workspace/ye/dataset/Dahoas/full_hh_rlhf \
   --num_padding_at_beginning 1 \
   --gradient_accumulation_steps 2 \
   --actor_learning_rate 1e-5 \
   --enable_ema \
   --actor_lora_dim 8 \
   --actor_lora_module_name decoder.layers. \
   --critic_lora_dim 8 \
   --critic_lora_module_name decoder.layers. \
   --print_answers_interval 10 \
   --print_answers \
   --critic_learning_rate 5e-6 \
   --max_prompt_seq_len 512 \
   --max_answer_seq_len 512 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --project_name debug_model \
   --run_name hh-rlhf-ppo\
   --enable_wandb \
   --deepspeed --enable_hybrid_engine --actor_gradient_checkpointing --actor_dropout 0.0 \
   --output_dir $OUTPUT 
#    &> $OUTPUT/full_hh_rlhf_ppo &