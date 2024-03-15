import os
import math
import sys
from shutil import copyfile

import torch
from transformers import (
    AutoTokenizer,
    AutoModel
)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import sys
sys.path.append('/workspace/ye/DPO')
sys.path.append('/workspace/ye')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from Config import DPO_config as cfgs, update_config
from utils import *
import time
from accelerate import Accelerator
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from Load_Data import *
from DPO_Engine import *
from loss import FocalLoss
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
accelerator = Accelerator()

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    parser.add_argument(
        '--config_file',
        type=str,
        default='',
        help="Path to training config file"
    )

    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")

    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    update_config(cfgs, args)
    mkdir(cfgs.log.output_dir)
    cfgs.seed = 42
    cfgs.is_local_main_process = accelerator.is_local_main_process
    if accelerator.is_local_main_process:
        init_tracker(cfgs)
        cfg_save_path = os.path.join(cfgs.log.output_dir, os.path.basename(args.config_file))
        copyfile(args.config_file, cfg_save_path)
    set_random_seed(cfgs.seed)
    accelerator.print('*************Load Tokenizer**************')
    actor_tokenizer = AutoTokenizer.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    if cfgs.model.model_type == 'opt':
        actor_tokenizer.pad_token_id = 0
        actor_tokenizer.eos_token_id = 2
    train_dataset,val_dataset,num_total_iters= create_datasets(cfgs)
    train_dataloader, eval_dataloader = create_dpo_dataloader(cfgs,actor_tokenizer, train_dataset, val_dataset)
    accelerator.print('*************Start DPO Engine**************')
    rlhf_engine = DPO_Engine(
                train_dataloader,
                eval_dataloader,
                actor_model_name_or_path=cfgs.model.model_path,
                actor_tokenizer=actor_tokenizer,
                num_total_iters=num_total_iters,
                cfgs=cfgs,accelerator=accelerator)

    total_steps = 0
    accelerator.print(f"-------------------Beginning Evaluation------------------")
    trainer = RLHFDPOTrainer(rlhf_engine, cfgs)
    best_score = evaluate_dpo(trainer, 
                    rlhf_engine.eval_dataloader, 
                    total_steps, 
                    accelerator.device, 
                    cfgs)
    accelerator.print(f"step:{total_steps}, losses:{best_score:.2f}")
    accelerator.print("***** Running training *****")
    best_score = -float('inf')
    losses= []
    for epoch in range(cfgs.train.num_train_epochs):
        accelerator.print(f"Beginning of Epoch {epoch+1}/{cfgs.train.num_train_epochs}, Total Micro Batches {len(train_dataloader)}")
        with tqdm(enumerate(train_dataloader),total = len(train_dataloader),desc=f"current epoch : {epoch}",\
                    disable=(not accelerator.is_local_main_process)) as pbar:
                for step ,batch_pairs in pbar:
                    batch_pairs = to_device(batch_pairs, accelerator.device)
                    loss, stats = trainer.train(batch_pairs)
                    # stats = {k: gather_objects(stats[k]) for k, v in stats.items()}
                    # stats = {k: sum(v)/len(v) for k, v in stats.items()}
                    total_steps += 1
                    losses.append(loss.item())

                    if total_steps % cfgs.log.eval_interval == 0:
                        mean_score = evaluate_dpo(trainer, val_dataloader=eval_dataloader, total_steps=total_steps, device=accelerator.device,cfgs=cfgs)
                        stats['losses/mean'] = sum(losses)/len(losses)

                        if mean_score > best_score: # Always save the best model
                            best_score = mean_score
                            if accelerator.is_local_main_process:
                                save_dpo_model(cfgs, rlhf_engine, actor_tokenizer, subffix=f'best')

                    if accelerator.is_local_main_process:
                        wandb_log(stats,total_steps)
    

def remove_file(save_dir,k=1):
    model_files = glob(os.path.join(save_dir, '*.pth'))
    model_files.sort(key=lambda x: float(x.split('-loss-')[1].split('.pth')[0]) if 'loss' in x else float('inf'))
    if len(model_files) > k:
        files_to_delete = model_files[k:]
        for file_to_delete in files_to_delete:
            os.remove(file_to_delete)

def get_test_result(save_path):
    pass
if __name__ == "__main__":
    main()