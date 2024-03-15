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
export PYTHONPATH=$PYTHONPATH:/workspace/ye/DPO
# deepspeed --include localhost:1,2
export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --config_file /workspace/ye/DPO/yaml/conf.yaml --main_process_port 29501\
                    /workspace/ye/DPO/main.py \
                --config_file /workspace/ye/DPO/yaml/DPO.yaml