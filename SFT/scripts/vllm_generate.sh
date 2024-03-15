export PYTHONPATH=$PYTHONPATH:/workspace/ye/SFT
NUM_GPUS=4
DATE_TODAY=304
MODEL_PATH="/workspace/ye/SFT/checkpoint/mistral/304-lr1e5-mistral/epoch-1"
checkpoint_model_path='/workspace/ye/SFT/checkpoint/mistral/304-lr1e5-mistral/epoch-0'
# '/tmp/dpo-gov-1e6-beta-0.1-ep1-1000step/checkpoint-1000'
#/workspace/ye/SFT/dataset/gov_data/valid.jsonl
# TEST_DATA_PATH="/workspace/ye/AI_FeedBack/data/oasst1/oasst1_prompts.json"
TEST_DATA_PATH='/workspace/ye/dataset/Gsm8k/test.jsonl'
SAVE_PATH="/workspace/ye/SFT/infer_result/${DATE_TODAY}"
num_return_sequences=1
DATA_TYPE=Gsm8K
TEMPERATRUE=0
NUM_SAMPLES=1500
mkdir -p $SAVE_PATH
# greedy search high data
CUDA_VISIBLE_DEVICES=0,1,2,3 python /workspace/ye/SFT/vllm_inference.py \
    --num_gpus $NUM_GPUS \
    --data_path $TEST_DATA_PATH \
    --init_model_path $MODEL_PATH \
    --checkpoint_model_path $checkpoint_model_path \
    --save_path ${SAVE_PATH}/Gsm8K-ep1.jsonl \
    --max_gen_length 1024 \
    --temperature $TEMPERATRUE \
    --num_return_sequences $num_return_sequences \
    --data_type $DATA_TYPE \
    --num_samples $NUM_SAMPLES \
    --style $DATA_TYPE