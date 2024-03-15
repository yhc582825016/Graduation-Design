import os
import time
import json
import argparse
import itertools

import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel , AutoConfig, AutoModelForCausalLM,LlamaForCausalLM,LlamaTokenizer,MistralForCausalLM
from datetime import datetime
# from loguru import logger
import sys
sys.path.append("/workspace/ye/SFT")
sys.path.append("/workspace/ye")
from accelerate import Accelerator
import math
from utils import *
from prompt import *
import random
import re
accelerator = Accelerator()
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--greedy_decode" , action= 'store_true')
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")

    parser.add_argument("--num_gpus", type=int, default=1, help='')
    parser.add_argument("--data_root", type=str, help='')
    parser.add_argument("--data_type", type=str, help='')
    parser.add_argument("--max_gen_len", type=int, help='')
    parser.add_argument("--nums_samples",type=int, help="选择生成的样本数量",default=None)
    parser.add_argument("--num_return_sequences", type=int, default=1, help='')
    parser.add_argument("--temperature", type=float, default=1.0, help='')

    parser.add_argument("--model_type", type=str, default='glm', help='')
    parser.add_argument("--sft_model_path", type=str, default=None, help='')
    parser.add_argument("--init_model_path", type=str, default=None, help='')

    parser.add_argument("--result_save_path", type=str, default=None, help='')
    parser.add_argument("--principle_path", type=str, default=None, help='')
    parser.add_argument("--prompt", type=str, default="", help='前缀提示词')
    parser.add_argument("--log_files", type=str, default=f"{datetime.now()}", help='log_path')
    parser.add_argument("--repeat_times", type=int, default=1, help='生成多少次')

    return parser.parse_args()

def Load_Sharegpt(data_path):
    dataset = load_json(data_path)
    meta_info = []
    prompt_list = []
    for data in dataset:
        prompt_list.append(data['prompt'])
        meta_info.append(data)
    return prompt_list, meta_info

def Load_Oasst1(data_path):
    dataset = load_json(data_path)
    meta_info = []
    prompt_list = []
    for data in dataset:
        prompt_list.append(data['instruction'])
        meta_info.append(data)
    return prompt_list, meta_info

def Load_rlhf(data_path,principle_path):
    dataset = load_jsonl(data_path)
    principles = load_json(principle_path)
    collected_dimension = []
    meta_info = []
    prompt_list = []
    principles_definition = ''
    for i in range(len(principles)):
        dimension = principles[i]['dimension']
        definition = principles[i]['definition']
        principles_definition += dimension + ":" + definition + "\n"
        collected_dimension.append(dimension)
    for idx in range(len(dataset)):
        history = ''
        labels = random.randint(1,2)
        for i in dataset[idx]['context']:
            history+=i['role']+":"+i['text']+"\n"
        chosen,rejected = dataset[idx]['chosen']['text'],dataset[idx]['rejected']['text']
        if labels == 1:
            prompt = MULTI_PROMPT_V0007.format(history,chosen,rejected,principles_definition,",".join(collected_dimension))
            Ground_Truth = "(a)"
        elif labels == 2:
            prompt = MULTI_PROMPT_V0007.format(history,rejected,chosen,principles_definition,",".join(collected_dimension))
            Ground_Truth = "(b)"
        prompt_list.append(prompt)
        meta_info.append({"chosen":chosen,'rejected':rejected,"Ground_Truth":Ground_Truth,'history':history})
    return prompt_list,meta_info

def Load_Gsm8K(data_path):
    dataset = load_json(data_path)
    meta_info = []
    prompt_list = []
    for data in dataset:
        prompt_list.append(f"{data['question']}Question:")
        meta_info.append(data)
    return prompt_list, meta_info

def Load_shuiwu(data_path):
    dataset = load_json(data_path)
    meta_info = []
    prompt_list = []
    for data in dataset:
        prompt_list.append(f"Question: {data['split_policy']}\n\nAnswer:")
        meta_info.append(data)
    return prompt_list, meta_info

def Load_dpo_data(data_path):
    dataset = load_json(data_path)
    meta_info = []
    prompt_list = []
    for data in dataset:
        prompt_list.append(f"Question: {data['question']}\n\nAnswer:")
        meta_info.append(data)
    return prompt_list, meta_info

def rename_model_params(state_dict):
    new_state_dict = {}
    for name, param in state_dict.items():
        if name.startswith('model.'):
            new_name = name.replace('model.', '')
            new_state_dict[new_name] = param
    return new_state_dict

class LLM_Inference:
    def __init__(self, args):
        self.args = args
        if args.model_type == 'chatglm': 
            self.tokenizer = AutoTokenizer.from_pretrained(args.init_model_path, trust_remote_code=True)  
            model = AutoModel.from_pretrained(args.init_model_path, trust_remote_code=True).eval()
        elif args.model_type == 'llama':
            self.tokenizer = LlamaTokenizer.from_pretrained(args.init_model_path, trust_remote_code=True)  
            model = LlamaForCausalLM.from_pretrained(args.init_model_path, trust_remote_code=True).eval()
            self.tokenizer.pad_token_id = 2
            self.tokenizer.eos_token_id = 2
        elif args.model_type == 'mistral':
            self.tokenizer = AutoTokenizer.from_pretrained(args.init_model_path, trust_remote_code=True)  
            model = MistralForCausalLM.from_pretrained(args.init_model_path, trust_remote_code=True).eval()
            self.tokenizer.pad_token_id = 2
            self.tokenizer.eos_token_id = 2
        if args.sft_model_path is not None:
            # if args.model_type == 'llama':
            #     model.load_state_dict(rename_model_params(torch.load(args.sft_model_path)))
            # else:
            model.load_state_dict(torch.load(args.sft_model_path))
        self.tokenizer.truncation_side = 'left'
        # self.model = deepspeed.init_inference(
        #     model=model,      # Transformers models
        #     mp_size=world_size,        # Number of GPU
        #     dtype=torch.float32, # dtype of the weights (fp16)
        #     replace_method="auto", # Lets DS autmatically identify the layer to replace
        #     replace_with_kernel_inject=True, # replace the model with the kernel injector
        #     max_out_tokens=2048
        # )
        self.model = accelerator.prepare_model(model,evaluation_mode=True)

    def generate_chatglm(self, batch, gen_kwargs):
        input_ids = self.tokenizer(batch, return_tensors="pt",  max_length=1024,truncation=True)
        input_ids = input_ids.to(torch.cuda.current_device())
        with torch.no_grad():
            outputs = self.model.generate(**input_ids, **gen_kwargs)

        return outputs[:, len(input_ids["input_ids"][0]):]
    
    def generate_llama(self, batch, gen_kwargs):
        # text = f"Question: {batch['question']}\n\nAnswer:"
        input_ids = self.tokenizer(batch, return_tensors="pt",  max_length=1024,truncation=True)
        input_ids = input_ids.to(torch.cuda.current_device())
        with torch.no_grad():
            outputs = self.model.generate(**input_ids, **gen_kwargs)
        return outputs[:, len(input_ids["input_ids"][0]):] 
    def generate_mistral(self, batch, gen_kwargs):
        # text = f"Question: {batch['question']}\n\nAnswer:"
        input_ids = self.tokenizer(batch, return_tensors="pt",  max_length=1024,truncation=True)
        input_ids = input_ids.to(torch.cuda.current_device())
        with torch.no_grad():
            outputs = self.model.generate(**input_ids, **gen_kwargs)
        return outputs[:, len(input_ids["input_ids"][0]):] 
    def process(self, prompts, gen_kwargs,rank):
        batch_size = 1 # TODO large bs
        response_list = []
        for i in tqdm(range(0, len(prompts), batch_size),disable=not rank!=0,desc="Current Generate"):
            # print_rank0(f"current in {i}-th samples/total {len(prompts)} samples",rank)
            sub_prompt = prompts[i:i + batch_size]
            data_info = {
                'response': []
                }
            try:
                for _ in range(args.repeat_times):
                    if self.args.model_type == 'chatglm':
                        outputs = self.generate_chatglm(sub_prompt, gen_kwargs)
                    elif self.args.model_type == 'llama':
                        outputs = self.generate_llama(sub_prompt, gen_kwargs)
                    elif self.args.model_type == 'mistral':
                        outputs = self.generate_mistral(sub_prompt, gen_kwargs)
                    else:
                        raise ValueError('{} not recognized.'.format(self.args.model_type))
                    for output in outputs.tolist():
                        output_str = self.tokenizer.decode(output[:], skip_special_tokens=True)
                        accelerator.print(output_str)
                        data_info["response"].append(output_str)
            except Exception as e:
                print(e)
                response_list.append(data_info)
                continue
            response_list.append(data_info)
        return response_list
    
if __name__ == '__main__':
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()
    def print_rank0(*msg):
        if rank != 0:
            return
        print(*msg)
    args = get_args()
    # if rank <=0: 
    #     logger.add(args.log_files,format="{name}{level}{message}",level="INFO",rotation="5 MB",encoding="utf-8")
    if rank == 0:
        with open(os.path.join(args.result_save_path, "config.json"),"w",encoding="utf-8") as f:
            json.dump(args.__dict__ , f , ensure_ascii = False , indent=4)
    infer_dtype = args.dtype
    # print_rank0(f"Using {world_size} gpus")
    model_name = args.model_type
    print_rank0(f"Loading model {model_name}")
    if args.greedy_decode:
        gen_kwargs = {
                    'do_sample': args.greedy_decode, 
                    'max_new_tokens': args.max_gen_len, 
                    'eos_token_id':2,
                    'num_return_sequences': args.num_return_sequences,
                    }
    else:
        gen_kwargs = {
            # 'top_k': args.top_k, 
            # 'top_p': args.top_p, 
            'temperature': args.temperature, 
            'max_new_tokens': args.max_gen_len, 
            'eos_token_id':2,
            'num_return_sequences': args.num_return_sequences,
            'do_sample':True
            }

    meta_info = None
    if args.data_type == 'sharegpt':
        prompt_list, meta_info = Load_Sharegpt(args.data_root)
    elif args.data_type == 'oasst1':
        prompt_list, meta_info = Load_Oasst1(args.data_root)
    elif args.data_type == 'hh_rlhf':
        prompt_list,meta_info = Load_rlhf(args.data_root,args.principle_path)
    elif args.data_type == 'Gsm8K':
        prompt_list,meta_info = Load_Gsm8K(args.data_root)
    elif args.data_type == 'gov':
        prompt_list,meta_info = Load_shuiwu(args.data_root)
    elif args.data_type == 'dpo':
        prompt_list,meta_info = Load_dpo_data(args.data_root)
    else:
        raise ValueError('{} not recognized.'.format(args.data_type))
    if args.nums_samples is not None:
        prompt_list = prompt_list[:args.nums_samples]
        meta_info = meta_info[:args.nums_samples]
    print_rank0(f"total samples num {len(prompt_list)}")
    print_rank0(f"*** Starting to generate {args.max_gen_len} tokens with bs={args.batch_size}, #prompts={len(prompt_list)}")
    print_rank0(f"Generate args {gen_kwargs}")

    glm_model = LLM_Inference(args)
    chunk_size = len(prompt_list) // args.num_gpus + 1
    data_dict = dict()
    meta_info_dict = dict()
    for i in range(args.num_gpus):
        data_dict[i] = prompt_list[i * chunk_size : (i + 1) * chunk_size] if i != args.num_gpus - 1 else prompt_list[i * chunk_size:] 
        if meta_info: #适配math数据
            meta_info_dict[i] = meta_info[i * chunk_size : (i + 1) * chunk_size] if i != args.num_gpus - 1 else meta_info[i * chunk_size:]
    st = time.time()
    response_list = glm_model.process(data_dict[rank], gen_kwargs,rank)
    
    if meta_info:
        try:
            for r, m in zip(response_list, meta_info_dict[rank]):
                r.update(m)
        except:
            print("m",m)
            print("r",r)
    et = time.time()
    save_json(os.path.join(args.result_save_path,f"rank-{rank}.json"), response_list)
    all_list = gather_objects(response_list)
    if rank == 0: 
        if args.data_type == 'hh_rlhf':
            correct_count = 0
            for i in all_list:
                # pattern = r'[Oo]utput \([Aa]\)|[Oo]utput \([Bb]\)'
                pattern = r'[Oo]utput \([ab]\)(?![\s\S]*[Oo]utput \([ab]\))'
                first_match = re.search(pattern, i['response'][0])
                match_result = first_match.group() if first_match else ''
                if i['Ground_Truth'] in match_result:
                    correct_flag =True
                    correct_count+=1
                else:
                    correct_flag = False
                i['correct_flag'] =  correct_flag
                i['match_result'] = match_result

            print(f'{args.data_type}-{args.model_type}-{args.nums_samples}-{correct_count/len(all_list):.2f}%:Acc')
            save_json(os.path.join(args.result_save_path,f'{correct_count/len(all_list)*100:.2f}%Acc.json'), all_list)
        elif args.data_type == 'Gsm8K':
            correct_count = 0
            try:
                for i in all_list:
                    if i['response'][0].split("#### ")[-1] == i['answer'].split("#### ")[-1]:
                        correct_count+=1
            except:
                print_rank0('something wrong')
            print(f'{correct_count/len(all_list)*100:.2f}%Acc')
            save_json(os.path.join(args.result_save_path,f'{correct_count/len(all_list)*100:.2f}%Acc.json'), all_list)
        else:
            save_json(os.path.join(args.result_save_path,"merged_json.json"), all_list)
    print('Rank: {}, Time cost: {} json saved in {}'.format(rank, et - st,args.result_save_path))
    # if rank<=0:
    #     logger.info('samples test Rank: {}, Time cost: {}'.format(rank, et - st))
