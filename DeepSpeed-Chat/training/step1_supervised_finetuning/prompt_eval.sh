export PYTHONPATH=$PYTHONPATH:/workspace/ye/DeepSpeed-Chat
python /workspace/ye/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py \
    --model_name_or_path_baseline  /workspace/ye/DeepSpeed-Chat/training/step1_supervised_finetuning/output \
    --model_name_or_path_finetune /workspace/ye/DeepSpeed-Chat/training/step3_rlhf_finetuning/actor \
    --max_new_tokens 256

