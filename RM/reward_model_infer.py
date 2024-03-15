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

accelerator = Accelerator(mixed_precision='bf16')

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

class CumstomModel(nn.Module):
    def __init__(self,model,args):
        nn.Module.__init__(self)
        self.args = args
        self.sentence_encoder = model
        self.batch_size = args.train.per_device_train_batch_size
        self.eval_batch_size = args.train.per_device_eval_batch_size
        self.linear = nn.Linear(model.config.hidden_size,1)
    
    def forward_chatglm(self,mode,**inputs):
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
            meta_info[i]['chosen_logit'] = chosen_logit.item()
            meta_info[i]['reject_logit'] = reject_logit.item()
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
    cfgs.seed = 42
    
    set_random_seed(cfgs.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    config = AutoConfig.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    # config.pre_seq_len = 10
    model = AutoModel.from_pretrained(cfgs.model.model_path,trust_remote_code=True,config=config)
    model = CumstomModel(model,cfgs)
    model.load_state_dict(torch.load(cfgs.evaluator.checkpoint_path))
    model.half()
    test_dataset = CustomDataset_For_Rankloss(cfgs.dataset.test_data_path, 
                                tokenizer,cfgs.dataset.max_src_len, cfgs.dataset.max_seq_len, model_type=cfgs.model.model_type,cfgs=cfgs)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                  collate_fn=DataCollatorWithPadding(tokenizer, cfgs.dataset.max_seq_len),
                                  sampler=test_sampler,
                                  batch_size=cfgs.train.per_device_eval_batch_size,
                                  pin_memory=True)
    model= accelerator.prepare_model(model)
    # test_dataloader = accelerator.prepare_data_loader(test_dataloader)

    accelerator.print(f"-------------------Beginning Evaluation-----------------")

    model.eval()
    all_info = []
    with tqdm(enumerate(test_dataloader),total = len(test_dataloader),disable=not accelerator.is_local_main_process) as pbar:
        for step,batch in pbar:
            batch = to_device(batch, accelerator.device)
            with torch.no_grad():
                # loss,chosen_logits,reject_logits = model('eval',**batch,return_dict=True
                loss,meta_info,bs_correct_count= model.forward_chatglm('eval',**batch,return_dict=True)
                all_info.extend(meta_info)
        # 
        datatype_dic={}
        for i in all_info:
            if i['dataset'] not in datatype_dic:
                datatype_dic[i['dataset']]={'loss':[i['loss']],'correctness':[i['correctness']],'chosen_logit':[i['chosen_logit']],'reject_logit':['reject_logit']}
            else:
                datatype_dic[i['dataset']]['loss'].append(i['loss'])
                datatype_dic[i['dataset']]['correctness'].append(i['correctness'])
                datatype_dic[i['dataset']]['chosen_logit'].append(i['chosen_logit'])
                datatype_dic[i['dataset']]['reject_logit'].append(i['reject_logit'])
        Metrics = {}
        for key in list(datatype_dic.keys()):
            correctness = datatype_dic[key]['correctness']
            loss = datatype_dic[key]['loss']
            Metrics[key]={'acc':sum(correctness)/len(correctness),'loss':sum(loss)/len(correctness)}
    hh_rlhf_loss = Metrics.get('hh_rlhf',{'loss':0})['loss']
    hh_rlhf_acc = round(Metrics.get('hh_rlhf',{"acc":0})['acc']*100,2)
    ultrafeedback_loss = Metrics.get('ultrafeedback',{'loss':0})['loss']
    ultrafeedback_acc = round(Metrics.get('ultrafeedback',{"acc":0})['acc']*100,2)
    accelerator.print(f"hh_rlhf losses:{hh_rlhf_loss}, acc: {hh_rlhf_acc}")
    accelerator.print(f"ultrafeedback losses:{ultrafeedback_loss}, acc: {ultrafeedback_acc}")
    all_info = gather_objects(all_info)
    accelerator.print(len(all_info))
    if accelerator.is_local_main_process:
        with open(os.path.join(cfgs.dataset.infer_result,cfgs.log.run_name+f"hh_acc:{hh_rlhf_acc}%-ultra_acc:{ultrafeedback_acc}%.json"),"w",encoding='utf-8') as f:
            json.dump(all_info,f,ensure_ascii=False,indent=4)

if __name__ == '__main__':
    main()