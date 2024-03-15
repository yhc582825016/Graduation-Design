export PYTHONPATH=$PYTHONPATH:/workspace/ye
# deepspeed --include localhost:1,2
export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
# pip install yacs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --config_file /workspace/ye/RM/yaml/config.yaml  --main_process_port 29502\
                    /workspace/ye/RM/reward_model.py \
                    --config_file /workspace/ye/RM/yaml/rm.yaml
# nohup bash /workspace/ye/AI_FeedBack/RM/run.sh > /workspace/ye/AI_FeedBack/RM/train.log > 2>&1 &