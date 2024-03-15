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

export PYTHONPATH=$PYTHONPATH:/workspace/ye/SFT
# SFT_MODEL_PATH='/workspace/ye/DPO/checkpoint/llama2/Gsm8K/merged_model/pytorch_model.bin'
# List of initial model paths
init_model_paths=('/checkpoint/304-lr1e5-mistral/epoch-2')
#/workspace/ye/llm/chatglm3-6b-base
#/workspace/ye/llm/llama2-7b
MODEL_TYPE='mistral'
date=304
NUM_GPUS=7
DATESTR=$(date +%Y%m%d-%H%M%S)
num_return_sequences=1
dataset_names=("Gsm8k_ep2")

for init_model_path in "${init_model_paths[@]}"; do
    for Dataset_name in "${dataset_names[@]}"; do
        # Create result save path
        RESULT_SAVE_PATH="/workspace/ye/SFT/infer_result/${date}/${Dataset_name}/" # Modified to include model checkpoint in path
        echo ${Dataset_name}
        DATA_ROOT=/workspace/ye/dataset/Gsm8k/test.jsonl
        if [ ! -d "$RESULT_SAVE_PATH" ]; then
            mkdir -p "$RESULT_SAVE_PATH"
        fi
        deepspeed --include localhost:0,1,2,3,4,5,7 --master_port=25000 /workspace/ye/SFT/infer.py \
                --model_type ${MODEL_TYPE} \
                --num_gpus ${NUM_GPUS} \
                --data_root ${DATA_ROOT} \
                --greedy_decode \
                --data_type Gsm8K \
                --max_gen_len 1024 \
                --num_return_sequences ${num_return_sequences} \
                --result_save_path ${RESULT_SAVE_PATH} \
                --init_model_path ${init_model_path} \
                --nums_samples 1500 \
                --log_files ${RESULT_SAVE_PATH} \
                | tee ${RESULT_SAVE_PATH}/logs.log
    done
done


# Corrected array definition
# init_model_paths=(640)

# Assuming dataset_names is defined correctly elsewhere
# dataset_names=('Dataset1' 'Dataset2')

# for init_model_path in "${init_model_paths[@]}"; do
#     for Dataset_name in "${dataset_names[@]}"; do
#         # Create result save path with corrected variable expansion
#         RESULT_SAVE_PATH="/workspace/ye/SFT/infer_result/${date}/${Dataset_name}/sft-${init_model_path}-dpo-128step-iter3-valid-set"
#         echo ${Dataset_name}
#         DATA_ROOT="/workspace/ye/MATH/dataset/MeraMath_test.json"
        
#         if [ ! -d "$RESULT_SAVE_PATH" ]; then
#             mkdir -p "$RESULT_SAVE_PATH"
#         fi

#         # Corrected command line to avoid unexpected argument error
#         deepspeed --include localhost:0,1,2,3,4,5,7 --master_port=25000 /workspace/ye/SFT/infer.py \
#                 --model_type ${MODEL_TYPE} \
#                 --num_gpus ${NUM_GPUS} \
#                 --data_root "${DATA_ROOT}" \
#                 --greedy_decode \
#                 --data_type dpo \
#                 --max_gen_len 2048 \
#                 --num_return_sequences ${num_return_sequences} \
#                 --result_save_path "${RESULT_SAVE_PATH}" \
#                 --init_model_path "/checkpoint/llama/ncp/checkpoint-${init_model_path}/dpo_iter3" \
#                 --nums_samples 1000 \
#                 --log_files "${RESULT_SAVE_PATH}" \
#                 | tee "${RESULT_SAVE_PATH}/logs.log"
#     done
# done
