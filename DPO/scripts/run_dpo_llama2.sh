# export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
# export WANDB_PROJECT=DPO
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --config_file /workspace/ye/DPO/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29502\
#                             /workspace/ye/DPO/dpo_llama2.py \
#                     --model_name_or_path /checkpoint/llama/ncp/checkpoint-160 \
#                     --data_path /workspace/ye/DPO/Dataset_NCP/DPO_iter1_step160 \
#                     --beta 0.1 \
#                     --num_train_epochs 1 \
#                     --learning_rate 5e-7 \
#                     --output_dir /checkpoint/llama/ncp/checkpoint-160/dpo_iter1 \
#                     --datatype gov_data \
#                     --run_name dpo_llama_5e-7_ep1


# 设置W&B API Key和项目名
export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
# export WANDB_PROJECT=DPO

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 循环执行checkpoint-160, 320, 480, 640
for step in 640
do
  # 更新model_name_or_path和data_path
  model_path="/checkpoint/llama/ncp/checkpoint-$step"
  data_path="/workspace/ye/DPO/Dataset_NCP/DPO_iter3_step$step"
  output_dir="/checkpoint/llama/ncp/checkpoint-$step/dpo_iter3"
  
  # 执行命令
  accelerate launch --config_file /workspace/ye/DPO/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29502 \
                    /workspace/ye/DPO/dpo_llama2.py \
                    --model_name_or_path $model_path \
                    --data_path $data_path \
                    --beta 1 \
                    --num_train_epochs 1 \
                    --learning_rate 5e-7 \
                    --output_dir $output_dir \
                    --datatype gov_data \
                    --run_name "dpo_llama_${step}_5e-7_ep1"
done

