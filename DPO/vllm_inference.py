import json
import argparse
import sys
sys.path.append('/workspace/ye')
import os
from vllm import LLM, SamplingParams
from transformers import LlamaTokenizer,AutoTokenizer
import torch
import pandas as pd
from datasets import load_dataset
def save_jsonL(save_path, result_list):
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(save_path, 'w',encoding='utf-8') as f:
        for item in result_list:
            json.dump(item, f,ensure_ascii=False)
            f.write('\n')

def save_json(save_path, json_obj):
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(save_path, 'w',encoding='utf-8') as f:
        json.dump(json_obj, f, indent=4,ensure_ascii=False)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=0.9, help="")
    parser.add_argument(
                        "--dtype",
                        type=str,
                        help="float16 or int8",
                        choices=["int8", "float16"],
                        default="float16",
                        )
    parser.add_argument("--presence_penalty", type=float, default=0.9)
    parser.add_argument("--frequency_penalty", type=float, default=0.9)
    parser.add_argument("--data_path", type=str, help="", required=True)
    parser.add_argument("--data_type", type=str, help="", default="sharegpt")
    parser.add_argument("--max_gen_length", type=int, help="", default=1024)
    parser.add_argument("--num_return_sequences", type=int, default=1, help="")
    parser.add_argument("--num_samples", type=int, default=None, help="")
    parser.add_argument("--checkpoint_model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None, help="")

    return parser.parse_args()


class LLMInference:
    def __init__(self, args):
        self.args = args
        self.data_type = args.data_type
        self.dataset = load_datasets(args.data_path, args.data_type)
        if self.args.num_samples is not None:
            self.dataset = self.dataset[:self.args.num_samples]
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_model_path, trust_remote_code=True)
        self.model = LLM(model=args.checkpoint_model_path,tensor_parallel_size=args.num_gpus,trust_remote_code=True)
        self.model.set_tokenizer = tokenizer
        self.gen_kwargs = {
            # "top_p": args.top_p,
            # "top_k": args.top_k,
            "temperature": args.temperature,
            "max_tokens": args.max_gen_length,
            "n": args.num_return_sequences,
            # "presence_penalty": args.presence_penalty,
            # "frequency_penalty": args.frequency_penalty,
            "skip_special_tokens": False,
            "stop_token_ids": [2],
        }
        self.sampling_params = SamplingParams(**self.gen_kwargs)


    def generate(self, batch):
        output_text = []
        outputs = self.model.generate(batch, self.sampling_params)
        for output in outputs:
            for i in range(self.args.num_return_sequences):
                print(output.outputs[i].text)
                output_text.append(output.outputs[i].text)
        return output_text

    def combine_prompt(self, sample):
        
        if self.data_type == "sharegpt":
            input_ = sample["prompt"]
        elif self.data_type == 'hh-rlhf':
            pass
        elif self.data_type == 'oasst1':
            input_ = sample["instruction"]
        elif self.data_type == 'Gsm8K':
            input_ = "Question:"+sample["question"]+'\n\nAssistant:'
        elif self.data_type == 'stack':
            input_ = "Question:"+sample["question"]+'\n\nAssistant:'
        elif self.data_type == 'goverment':
            input_ = "Question:"+sample["question"]+'\n\nAnswer:'
        else:
            raise ValueError(f"{self.data_type} not recognized.")
        return input_

    def process(self):
        data_list = self.dataset
        # result_list = []
        sub_inputs = list(map(self.combine_prompt, data_list))
        output_str_list = self.generate(sub_inputs)

        for idx in range(len(data_list)):
            data_list[idx].update(
                {
                    "response": output_str_list[
                        idx
                        * self.args.num_return_sequences : (idx + 1)
                        * self.args.num_return_sequences
                    ]
                }
            )
        return data_list

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
    
def save_json(save_path, data):
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_jsonL(json_path):

    with open(json_path, 'r') as f:
        data = f.readlines()

    data = [json.loads(d) for d in data]

    return data

def save_jsonL(save_path, data):
    with open(save_path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def load_datasets(data_path, data_type):
    if data_type == "sharegpt":
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif data_type == "oasst1":
        data = load_json(data_path)
    elif data_type == "Gsm8K":
        data = load_jsonL(data_path)
    elif data_type == 'stack':
        data = load_dataset(data_path)['train']['question']
        data = [{"question":i} for i in data]
    elif data_type == 'goverment':
        data = load_dataset(data_path)['train']['question']
        data = [{"question":i} for i in data]
    else:
        raise ValueError("{} not recognized.".format(data_type))
    return data


if __name__ == "__main__":
    args = get_args()
    engine = LLMInference(args)

    print(
        f"*** Starting to generate {args.max_gen_length}, prompts={len(engine.dataset)}",
    )
    result_list = engine.process()
    save_json(args.save_path, result_list)
