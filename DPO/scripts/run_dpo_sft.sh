export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
export WANDB_PROJECT=DPO
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 accelerate launch --config_file /workspace/ye/DPO/accelerate_configs/deepspeed_zero2.yaml --main_process_port 25001 \
                            /workspace/ye/DPO/sft_llama2.py \
                            --learning_rate 1e-5 \
                            --max_steps 900 \
                            --dataset_name /workspace/ye/DPO/Dataset_NCP/SFT \
                            --split train \
                            --subset finetune \
                            --output_dir '/checkpoint/llama/ncp' \
                            --run_name goverment_retrive_1000setp \
                            # --num_train_epochs 1 \