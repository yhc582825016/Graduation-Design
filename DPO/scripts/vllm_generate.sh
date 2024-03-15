export PYTHONPATH=$PYTHONPATH:/workspace/ye/DPO
NUM_GPUS=4
DATE_TODAY=208
# base_model='/workspace/ye/llm/llama2-7b'
checkpoint_model_path='/workspace/ye/DPO/checkpoint/llama2/goverment_6000step_que2res/merged_model'
# TEST_DATA_PATH="/workspace/ye/AI_FeedBack/data/oasst1/oasst1_prompts.json"
TEST_DATA_PATH='/workspace/ye/DPO/Dataset/goverment_qa/finetune-que2res'
# TEST_DATA_PATH=/workspace/ye/dataset/stack-exchange-paired/evaluation

SAVE_PATH="/workspace/ye/DPO/infer_result/${DATE_TODAY}"
SAMPLE_NUM=1
DATA_TYPE=goverment
TEMPERATRUE=0
NUM_SAMPLES=1500
mkdir -p $SAVE_PATH
# greedy search high data
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python /workspace/ye/DPO/vllm_inference.py \
    --num_gpus $NUM_GPUS \
    --data_path $TEST_DATA_PATH \
    --checkpoint_model_path $checkpoint_model_path \
    --save_path ${SAVE_PATH}/${DATA_TYPE}-random-sample-temp${TEMPERATRUE}-${NUM_SAMPLES}.json \
    --max_gen_length 1024 \
    --temperature $TEMPERATRUE \
    --num_return_sequences $SAMPLE_NUM \
    --data_type $DATA_TYPE \
    --num_samples $NUM_SAMPLES \
    # --base_model $base_model