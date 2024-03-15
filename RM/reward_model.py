import argparse
import os
import math
import sys
from shutil import copyfile
from utils import *
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    SchedulerType,
    get_scheduler,
    Trainer
)
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from config import rm_config as cfgs, update_config
from accelerate import Accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import torch.nn as nn
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

accelerator = Accelerator()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

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

class model_args:
    hidden_dropout_prob = 0.1
    pre_seq_len = 20
    prefix_hidden_size = 512
    model_name_or_path = "bert-base-chinese"
    
model_path_map = {
    'bert':'bert-base-chinese',
    'bert_large':'/home/common_llm/chinese-roberta-wwm-ext-large',
    'chatglm3':'/home/common_llm/chatglm3-6b',
    'gpt2':'/home/common_llm/gpt2-chinese-cluecorpussmall'
}
class CumstomModel(nn.Module):
    def __init__(self,model,args):
        nn.Module.__init__(self)
        self.args = args
        self.sentence_encoder = model
        self.batch_size = args.train.per_device_train_batch_size
        self.eval_batch_size = args.train.per_device_eval_batch_size
        self.linear = nn.Linear(model.config.hidden_size,1)
    
    def forward(self,**inputs):
        meta_info = inputs['meta_info']
        chosen_input_ids = inputs["chosen_input_ids"]
        rejected_input_ids = inputs["rejected_input_ids"]
        input_ids = torch.cat([chosen_input_ids,rejected_input_ids],dim=0)
        bs = input_ids.size(0)//2
        sentence_logits = self.linear(self.sentence_encoder.transformer(input_ids=input_ids)[0].permute(1, 0, 2).contiguous())
        chosen_logits = sentence_logits[:bs]
        reject_logits= sentence_logits[bs:]
        loss = 0
        bs_correct_count = 0
        for i in range(bs):
            chosen_end_ids = torch.where(chosen_input_ids[i]==2)[0][0]
            reject_end_ids = torch.where(rejected_input_ids[i]==2)[0][0]
            chosen_logit = chosen_logits[i,chosen_end_ids,:].squeeze(-1)
            reject_logit = reject_logits[i,reject_end_ids,:].squeeze(-1)
            loss += -F.logsigmoid(chosen_logit - reject_logit)##bs*seq
            if chosen_logit > reject_logit:
                meta_info[i]['correctness'] = 1
                bs_correct_count+=1
            else:
                meta_info[i]['correctness'] = 0
            meta_info[i]['loss'] = loss.mean().item()
        return loss.mean(),meta_info,bs_correct_count
        

def main():
    args = parse_args()
    update_config(cfgs, args)
    mkdir(cfgs.log.output_dir)
    cfgs.seed = 42

    if accelerator.is_local_main_process:
        init_tracker(cfgs)
        cfg_save_path = os.path.join(cfgs.log.output_dir, os.path.basename(args.config_file))
        copyfile(args.config_file, cfg_save_path)

    set_random_seed(cfgs.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    config = AutoConfig.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    # config.pre_seq_len = 10
    # config.prefix_projecion = False
    # config.use_cache = False

    model = AutoModel.from_pretrained(cfgs.model.model_path,trust_remote_code=True,config=config)
    model = CumstomModel(model,cfgs)
    # model = model.half()
    # model.sentence_encoder.transformer.prefix_encoder.float()

    optimizer = torch.optim.AdamW(model.parameters(), cfgs.train.learning_rate)
    train_dataset = CustomDataset_For_Rankloss(cfgs.dataset.train_data_path, 
                                tokenizer,cfgs.dataset.max_src_len, cfgs.dataset.max_seq_len, model_type=cfgs.model.model_type,cfgs=cfgs)
    val_dataset = CustomDataset_For_Rankloss(cfgs.dataset.val_data_path, 
                                tokenizer,cfgs.dataset.max_src_len, cfgs.dataset.max_seq_len, model_type=cfgs.model.model_type,cfgs=cfgs)
    train_dataloader, eval_dataloader = create_rm_dataloder(cfgs, train_dataset, val_dataset, tokenizer)
        # scheduler
    if cfgs.train.scheduler == "CAWR":
        T_mult = cfgs.train.T_mult
        rewarm_epoch_num = cfgs.train.rewarm_epoch_num
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         int(len(train_dataloader)/accelerator.num_processes * rewarm_epoch_num),
                                                                         T_mult)
    elif cfgs.train.scheduler == "Step":
        decay_rate = cfgs.train.decay_rate
        decay_steps =  cfgs.train.decay_steps
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    else:    
        lr_scheduler = OneCycleLR(optimizer, max_lr=cfgs.train.learning_rate, epochs=cfgs.train.num_train_epochs, steps_per_epoch=len(train_dataloader))

    model,lr_scheduler,optimizer,train_dataloader,eval_dataloader= accelerator.prepare(model,lr_scheduler,optimizer,train_dataloader,eval_dataloader)
    global_step = 0
    accelerator.print(f"-------------------Beginning Evaluation-----------------")
    Metrics= evaluate_model_version(model, eval_dataloader,accelerator,cfgs)
    accelerator.print(f"step:{global_step}, hh_rlhf losses:{Metrics['hh_rlhf']['loss']}, acc: {round(Metrics['hh_rlhf']['acc']*100,2)}")
    accelerator.print(f"step:{global_step}, ultrafeedback losses:{Metrics['ultrafeedback']['loss']}, acc: {round(Metrics['ultrafeedback']['acc']*100,2)}")
    eval_stats = {
        'eval/hh_rlhf_loss':Metrics['hh_rlhf']['loss'] ,
        'eval/ultrafeedback_loss':Metrics['ultrafeedback']['loss'] ,
        'eval/hh_rlhf_acc': round(Metrics['hh_rlhf']['acc']*100,2),
        'eval/ultrafeedback_acc': round(Metrics['ultrafeedback']['acc']*100,2)
    }
    if accelerator.is_local_main_process:
        wandb_log(eval_stats, global_step)
        file_path = os.path.join(cfgs.train.save_path,cfgs.model.model_type,cfgs.log.run_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
    accelerator.print("***** Running training *****")
    best_loss = 10000000
    for epoch in range(cfgs.train.num_train_epochs):
        correct_count = 0
        total_count = 0
        accelerator.print(f"Beginning of Epoch {epoch+1}/{cfgs.train.num_train_epochs}, Total Micro Batches {len(train_dataloader)}")
        model.train()
        with tqdm(enumerate(train_dataloader),total = len(train_dataloader),desc=f"current epoch : {epoch}",\
                disable=(not accelerator.is_local_main_process)) as pbar:
            for step ,batch in pbar:
                batch = to_device(batch, accelerator.device)
                loss,meta_info,bs_correct_count = model.module.forward_chatglm('train',**batch)
                correct_count+=bs_correct_count
                total_count+=batch['chosen_input_ids'].size(0)
                acc = correct_count/total_count
                accelerator.backward(loss)
                pbar.set_description("loss:%.2f,acc:%.2f%%" % (loss, acc*100))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                stats = {'train/loss': loss.item(),
                        'train/epoch':epoch,
                        'train/acc':acc,
                        'lr': optimizer.param_groups[0]['lr']}
                global_step += 1
                if accelerator.is_local_main_process:
                    wandb_log(stats, global_step)
                if global_step % int((cfgs.log.eval_epoch_ratio*len(train_dataloader))) == 0:
                    # Evaluate perplexity on the validation set.
                    accelerator.print(f"***** Evaluating loss,  {global_step}/{cfgs.train.num_train_epochs * len(train_dataloader)} *****")
                    Metrics = evaluate_model_version(model, eval_dataloader,accelerator,cfgs)
                    eval_stats = {
                        'eval/hh_rlhf_loss':Metrics['hh_rlhf']['loss'] ,
                        'eval/ultrafeedback_loss':Metrics['ultrafeedback']['loss'] ,
                        'eval/SHEPHERD_loss':Metrics['SHEPHERD']['loss'] ,
                        'eval/hh_rlhf_acc': round(Metrics['hh_rlhf']['acc']*100,2) ,
                        'eval/ultrafeedback_acc': round(Metrics['ultrafeedback']['acc']*100,2),
                        'eval/SHEPHERD_acc': round(Metrics['SHEPHERD']['acc']*100,2)
                    }
                    accelerator.print(f"step:{global_step}, hh_rlhf losses:{Metrics['hh_rlhf']['loss']}, acc: {round(Metrics['hh_rlhf']['acc']*100,2)}")
                    accelerator.print(f"step:{global_step}, ultrafeedback losses:{Metrics['ultrafeedback']['loss']}, acc: {round(Metrics['ultrafeedback']['acc']*100,2)}")
                    accelerator.print(f"step:{global_step}, SHEPHERD losses:{Metrics['SHEPHERD']['loss']}, acc: {round(Metrics['SHEPHERD']['acc']*100,2)}")
                    if accelerator.is_local_main_process:
                        wandb_log(eval_stats, global_step)

                        if Metrics['hh_rlhf']['loss'] < best_loss:
                            best_loss = Metrics['hh_rlhf']['loss']
                            unwrapped_model = accelerator.unwrap_model(model)
                            
                            accelerator.print('save_model')
                            torch.save(unwrapped_model.state_dict(), os.path.join(file_path,f'epoch{epoch}-eval-loss-{best_loss:.2f}.pth'))
                            accelerator.print('save_model_finshed')
                            remove_file(file_path)
                    model.train()

def remove_file(save_dir,k=1):

    model_files = glob(os.path.join(save_dir, '*.pth'))
    model_files.sort(key=lambda x: float(x.split('-loss-')[1].split('.pth')[0]) if 'loss' in x else float('inf'))
    if len(model_files) > k:
        files_to_delete = model_files[k:]
        for file_to_delete in files_to_delete:
            os.remove(file_to_delete)

if __name__ == "__main__":
    main()