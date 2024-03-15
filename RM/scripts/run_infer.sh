export PYTHONPATH=$PYTHONPATH:/workspace/ye
# deepspeed --include localhost:1,2
# pip install yacs
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file /workspace/ye/RM/yaml/config.yaml \
                    /workspace/ye/RM/reward_model.py \
                    --config_file /workspace/ye/RM/yaml/rm.yaml
# nohup bash /workspace/ye/AI_FeedBack/RM/run.sh > /workspace/ye/AI_FeedBack/RM/train.log > 2>&1 &