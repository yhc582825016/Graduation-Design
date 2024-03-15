import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    OPTForCausalLM
)
from utils import log_init,get_optimizer_grouped_parameters
from torch.optim.lr_scheduler import OneCycleLR,LinearLR
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from utils import *
import torch.nn as nn
import torch.nn.functional as F

class DPO_Engine():
    def __init__(self,train_dataloader,eval_dataloader,actor_model_name_or_path,
                 actor_tokenizer, cfgs, num_total_iters,accelerator):
        self.cfgs = cfgs
        self.num_total_iters = num_total_iters
        self.actor_tokenizer = actor_tokenizer
        self.actor_model_type = cfgs.model.model_type
        if cfgs.model.model_type == 'chatglm':
            self.model_class = AutoModel
        elif cfgs.model.model_type == 'opt':
            self.model_class = OPTForCausalLM
        # Model
        actor_model = self.model_class.from_pretrained(actor_model_name_or_path,trust_remote_code=True)
        ref_model = self.model_class.from_pretrained(actor_model_name_or_path,trust_remote_code=True)
        # Optimizer
        optim_params = get_optimizer_grouped_parameters(actor_model,cfgs.train.weight_decay)
        optimizer = torch.optim.AdamW(optim_params,lr=cfgs.train.learning_rate,betas=(0.9, 0.95))
        lr_scheduler = OneCycleLR(optimizer, max_lr=cfgs.train.learning_rate, epochs=cfgs.train.num_train_epochs, steps_per_epoch=num_total_iters)
        self.actor,self.lr_scheduler,self.optimizer,self.train_dataloader,self.eval_dataloader= \
            accelerator.prepare(actor_model,lr_scheduler,optimizer,train_dataloader,eval_dataloader)
        self.ref = accelerator.prepare_model(ref_model,evaluation_mode=True)

class RLHFDPOTrainer:

    def __init__(self, rlhf_engine, cfgs, config=None):

        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.ref_model = self.rlhf_engine.ref
        self.actor_tokenizer = self.rlhf_engine.actor_tokenizer
        self.ppl_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.cfgs = cfgs
        self.config = config
        self.label_pad_token_id = self.actor_tokenizer.pad_token_id
        self.beta = cfgs.train.beta


    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        # """
        # max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        device = batch['chosen_input_ids'].device
        concatenated_batch = {}
        concatenated_batch['concatenated_input_ids'] = torch.cat(
            (batch["chosen_input_ids"], batch["rejected_input_ids"]), dim=0
        ).to(device)
        concatenated_batch['concatenated_labels'] = torch.cat(
            (batch["chosen_labels"], batch["rejected_labels"]), dim=0
        ).to(device)
        concatenated_batch['concatenated_attention_mask'] = torch.cat(
            (batch["chosen_attention_mask"], batch["rejected_attention_mask"]), dim=0
        ).to(device)
        # s = batch["chosen_input_ids"].shape[-1]
        # bz = batch["chosen_input_ids"].shape[0]
        # concatenated_attention_mask = torch.ones(bz,1,2*s,2*s, dtype=torch.bool)
        # concatenated_attention_mask[:,:,s:,s:] = batch["chosen_attention_mask"]
        # concatenated_attention_mask[:,:,s:,s:] = batch["rejected_attention_mask"]
        # concatenated_batch['concatenated_attention_mask'] = concatenated_attention_mask.to(device)
        return concatenated_batch
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
    
    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = gather_log_probs(logits, labels)
        # per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
    
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)   
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        )
        all_logits = all_logits.logits.to(torch.float32)
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]
        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = all_logits[batch["chosen_input_ids"].shape[0] :]
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def train(
            self,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
        ):
            """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
            if train_eval == "train":
                self.actor_model.train()
            else:
                self.actor_model.eval()
            metrics = {}
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = self.concatenated_forward(self.actor_model, batch)
            with torch.no_grad():
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)
            
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            dpo_loss = losses.mean()
            if train_eval == "train":
                self.actor_model.backward(dpo_loss)
                self.actor_model.step()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            # print_rank_0(f"reward_accuracies: {reward_accuracies.mean()}, chosen{chosen_rewards}, reject{rejected_rewards}", self.cfgs.global_rank)

            if train_eval == "eval":
                prefix = "eval_"
                metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.detach().cpu().numpy().mean()
                metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).detach().cpu().numpy().mean()
            else:
                prefix = ""
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.detach().cpu().numpy().mean()
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.detach().cpu().numpy().mean()
            metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().numpy().mean()
            metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
            metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().numpy().mean()
            metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()
            metrics[f"{prefix}losses"] = dpo_loss.detach().cpu().item()

            return dpo_loss, metrics
    
    def evaluate(self, eval_dataloader, device,cfgs):
        eva_metrics = {}
        with tqdm(enumerate(eval_dataloader),total = len(eval_dataloader),\
                    disable=(not cfgs.is_local_main_process)) as pbar:
            for step ,batch_pairs in pbar:
                with torch.no_grad():
                    torch.distributed.barrier()
                    batch_pairs = to_device(batch_pairs, device)
                    _, metrics = self.train(batch_pairs, train_eval="eval")
                for k,v in metrics.items():
                    if k not in eva_metrics:
                        eva_metrics[k] = []
                    eva_metrics[k].append(v)

        return eva_metrics