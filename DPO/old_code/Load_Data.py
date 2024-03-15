from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
from transformers import PreTrainedTokenizerBase
import json
from dataclasses import dataclass
import argparse
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
class DPODataset(Dataset):
    def __init__(self, args, file_path):
        """hh-rlhf gsm8k data
        args: 
            args
            file_path
        Returns:
            prompts,chosens,rejecteds
        """
        self.args = args
        if args.dataset.datatype=='hh-rlhf':
            self.prompt_list, self.chosen_list, self.rejected_list = self.load_data(file_path)
        elif args.dataset.datatype=='gsm8k':
            self.prompt_list, self.chosen_list, self.rejected_list = self.load_gsm8k_data(file_path)

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):

        prompt = self.prompt_list[idx]
        if "chatglm" in self.args.model.model_type:
            prompt = prompt.replace('[gMASK]', '').replace('[UNUSED1]', '').replace('[UNUSED2]', '')
        return {
            "prompt": prompt,
            "chosen": self.chosen_list[idx],
            "rejected": self.rejected_list[idx]
        }

    def load_data(self,file_path):
        with open(file_path,'r') as f:
            data = [json.loads(i) for i in f.readlines()]
        prompts = []
        chosens = []
        rejecteds = []
        history = ''
        for idx in data:
            history = [i['role'] + ":" + i['text'] for i in idx['context']]
            history_str = "\n".join(history)
            chosen, rejected = idx['chosen']['text'], idx['rejected']['text']
            
            prompts.append(history_str)
            chosens.append(chosen)
            rejecteds.append(rejected)
        return prompts,chosens,rejecteds
    
    def load_gsm8k_data(self,file_path):
        with open(file_path,'r') as f:
            data = json.load(f)
        prompts = []
        chosens = []
        rejecteds = []
        for idx in data:
            prompts.append(idx['question'])
            chosens.append(idx['answer'])
            rejecteds.append(idx['reject'])
        return prompts,chosens,rejecteds

def create_datasets(cfgs):
    train_dataset = DPODataset(cfgs, cfgs.dataset.train_data_path)
    val_dataset = DPODataset(cfgs, cfgs.dataset.val_data_path)
    iters_prompt = len(train_dataset) // cfgs.train.per_device_train_batch_size // torch.distributed.get_world_size()
    num_update_steps_per_epoch = iters_prompt * \
                                 cfgs.train.per_device_train_batch_size * \
                                 cfgs.train.num_train_epochs / cfgs.train.gradient_accumulation_steps
    num_total_iters = int(cfgs.train.num_train_epochs * num_update_steps_per_epoch)

    return train_dataset, val_dataset,  num_total_iters


def create_dpo_dataloader(cfgs, tokenizer, train_dataset, val_dataset):
    """Create dataloader for RLHF training
    """

    prompt_train_sampler = DistributedSampler(train_dataset)
    prompt_val_sampler = DistributedSampler(val_dataset,shuffle=True)
    
    prompt_train_dataloader = DataLoader(
        train_dataset,
        sampler=prompt_train_sampler,
        collate_fn=DPODataCollatorWithPadding(tokenizer,
                                              cfgs, 
                                              max_length=cfgs.dataset.max_seq_len, 
                                              max_prompt_length=cfgs.dataset.max_prompt_length, 
                                              max_response_length=cfgs.dataset.max_response_length, 
                                              label_pad_token_id=tokenizer.pad_token_id),
                                            batch_size=cfgs.train.per_device_train_batch_size,pin_memory=True)

    prompt_val_dataloader = DataLoader(
        val_dataset,
        sampler=prompt_val_sampler,
        collate_fn=DPODataCollatorWithPadding(tokenizer,
                                              cfgs, 
                                              max_length=cfgs.dataset.max_seq_len, 
                                              max_prompt_length=cfgs.dataset.max_prompt_length, 
                                              max_response_length=cfgs.dataset.max_response_length,  
                                              label_pad_token_id=tokenizer.pad_token_id),
                                            batch_size=cfgs.train.per_device_train_batch_size,pin_memory=True)
    return prompt_train_dataloader, prompt_val_dataloader

@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    tokenizer: PreTrainedTokenizerBase
    args: argparse.Namespace = None
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_response_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        # assert self.tokenizer.padding_side == "left", "In RLHF training, need to set padding_side to 'left'"
        assert self.tokenizer.eos_token not in prompt, f"Prompt contains EOS token: {prompt}"
        assert self.tokenizer.eos_token not in chosen, f"Chosen response contains EOS token: {chosen}"
        assert self.tokenizer.eos_token not in rejected, f"Rejected response contains EOS token: {rejected}"
        
        # prompt_tokens_tokenized = self.tokenizer.tokenize(prompt)
        # chosen_tokens_tokenized = self.tokenizer.tokenize(chosen)
        # rejected_tokens_tokenized = self.tokenizer.tokenize(rejected)
        prompt_tokens = prompt
        chosen_tokens = chosen
        rejected_tokens = rejected

        longer_response_length = max(len(chosen), len(rejected)) + 2

        # if combined sequence is too long, truncate the prompt
        if longer_response_length > self.max_length - self.max_prompt_length:
            # assert False
            if self.truncation_mode == "keep_start":
                prompt_tokens = prompt[: self.max_prompt_length]
            elif self.truncation_mode == "keep_end":
                prompt_tokens = prompt[-self.max_prompt_length :]
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
            
        # if that's still too long, truncate the response
        if len(prompt) + longer_response_length > self.max_length:
            # assert False
            chosen_tokens = chosen[: - self.max_response_length]
            rejected_tokens = rejected[: - self.max_response_length]

        # chosen_tokens = prompt_tokens + self.tokenizer.eos_token + chosen_tokens
        # rejected_tokens = prompt_tokens + self.tokenizer.eos_token + rejected_tokens

        # chosen_input_ids = self.tokenizer(prompt_tokens,chosen_tokens, 
        #                                   max_length=self.max_length, 
        #                                   truncation=True,
        #                                   padding=False)['input_ids']
        # rejected_input_ids = self.tokenizer(prompt_tokens,rejected_tokens,
        #                                   max_length=self.max_length, 
        #                                   truncation=True,
        #                                   padding=False)['input_ids']

        batch = {}
        chosen_input_ids,chosen_labels = self.get_ids(prompt_tokens,chosen_tokens)
        rejected_input_ids,rejected_labels = self.get_ids(prompt_tokens,rejected_tokens)
        # batch["prompt"] = prompt
        # batch["chosen"] = prompt + chosen
        # batch["rejected"] = prompt + rejected
        # batch["chosen_response_only"] = chosen
        # batch["rejected_response_only"] = rejected
        batch["chosen_input_ids"] = chosen_input_ids
        batch["rejected_input_ids"] = rejected_input_ids
        batch["chosen_labels"] = chosen_labels
        batch["rejected_labels"] = rejected_labels

        return batch
    
    def collate(self,  inputs):
        max_len = max([len(k['chosen_input_ids']) for k in inputs]+
                          [len(k['rejected_input_ids']) for k in inputs])
        batch = {}
        batch["chosen_input_ids"] = torch.tensor([x['chosen_input_ids'] +[self.tokenizer.pad_token_id] * (max_len - len(x['chosen_input_ids'])) for x in inputs]).view(-1, max_len)
        batch["chosen_attention_mask"] = torch.tensor([len(x['chosen_input_ids'])*[1] +[self.tokenizer.pad_token_id] * (max_len - len(x['chosen_input_ids'])) for x in inputs]).view(-1, max_len)
        batch["rejected_input_ids"]= torch.tensor([x['rejected_input_ids'] +[self.tokenizer.pad_token_id] * (max_len - len(x['rejected_input_ids'])) for x in inputs]).view(-1, max_len)
        batch["rejected_attention_mask"] = torch.tensor([len(x['rejected_input_ids'])*[1] +[self.tokenizer.pad_token_id] * (max_len - len(x['rejected_input_ids'])) for x in inputs]).view(-1, max_len)
        batch["chosen_labels"]  = torch.tensor([x['chosen_labels'] +[0] * (max_len - len(x['chosen_labels'])) for x in inputs]).view(-1, max_len)
        batch["rejected_labels"] = torch.tensor([x['rejected_labels'] +[0] * (max_len - len(x['rejected_labels'])) for x in inputs]).view(-1, max_len)

        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)
    def get_ids(self,prompt_tokens,response_tokens):
        a_ids = self.tokenizer.encode(prompt_tokens, add_special_tokens=True, truncation=True,
                                      max_length=self.max_prompt_length)
        b_ids = self.tokenizer.encode(response_tokens, add_special_tokens=False, truncation=True,
                                      max_length=self.max_response_length)
        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [0] * context_length + b_ids + [self.tokenizer.eos_token_id]
        return input_ids,labels
