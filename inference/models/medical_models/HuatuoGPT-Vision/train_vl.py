import os
import copy
import json
import torch
import logging
import argparse
import random
import shutil
from typing import List, Dict, Any
from dataclasses import dataclass

import wandb
import datasets
from tqdm import tqdm
import gc
from jinja2 import Template
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    set_seed,
    get_cosine_schedule_with_warmup
)
from torch.optim.lr_scheduler import LambdaLR
import math
from qwen_vl_utils import fetch_image

# Set environment variables
os.environ['WANDB_DISABLE_CODE'] = 'true'
os.environ["WANDB_API_KEY"] = ''
os.umask(0)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

def get_custom_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    min_lr_ratio=0.0, 
    num_cycles=0.5
):
    """
    Custom cosine scheduler supporting minimum learning rate ratio
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_learning_rate(step, initial_lr, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles=0.5):
    if step < num_warmup_steps:
        return float(step) / float(max(1, num_warmup_steps)) * initial_lr
    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
    scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
    return scaled_factor * initial_lr

class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss

class SftDataset(Dataset):
    """Pretraining dataset class"""
    def __init__(self, config, processor, accelerator):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        # Load pretraining data (supports multiple sources)
        self.data = []
        with open(config.data_path,'r') as f:
            data = json.load(f)
        self.data = data

        # Convert conversation format
        role_map = {'system': 'system', 'human': 'user', 'gpt': 'assistant'}
        if 'value' in data[0]['conversations'][0]:
            for da in data:
                conv = []
                for x in da['conversations']:
                    conv.append({
                        'role': role_map[x['from']],
                        'content': x['value']
                    })
                da['conversations'] = conv

        # Filter long samples and ensure assistant response is valid
        tmp_data = []
        for da in self.data:
            if len(da['image']) > 8:
                continue
            if da['conversations'][-1]['role'] != 'assistant' or len(da['conversations'][-1]['content'].strip()) == 0:
                continue
            tmp_data.append(da)
        self.data = tmp_data
        accelerator.print(f'Training data size: {len(self.data)} samples')

        self.assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.assistant_prefix_ids = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False, return_tensors="pt")[0]
        self.eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.image_sizes = (config.max_width, config.max_height)
        accelerator.print(f'Total dataset size: {len(self.data)}')

        # Padding
        self.image_padding = {'pixel_values': torch.zeros(24, 1176), 'image_grid_thw': torch.tensor([[1, 4, 6]])}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []

        for item in batch:
            img_num = 0

            if len(item['image']) > 0 and '<image>' not in ''.join([x['content'] for x in item['conversations']]):
                item["conversations"][0]['content'] = '\n'.join(['<image>']*len(item['image']))+'\n'+item["conversations"][0]['content']

            if len(item['image']) > 0:
                for k in item["conversations"]:
                    img_num +=  k['content'].count('<image>')
                    k['content'] = k['content'].replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
                assert img_num == len(item['image']), f"Image count mismatch: {img_num} != {len(item['image'])}\n{item['conversations']}"
            input_text = self.processor.apply_chat_template(item["conversations"], add_generation_prompt=False, tokenize=False)
            texts.append(input_text)
            images.extend(item["image"])

        img_list = [fetch_image({"type": "image", "image": img_path,"max_pixels": self.image_sizes[0] * self.image_sizes[1]}) for img_path in images]

        if len(img_list) > 0:
            inputs = self.processor(text=texts, images=img_list, return_tensors="pt", padding=True)
            pixel_values = inputs['pixel_values']
            image_grid_thw = inputs['image_grid_thw']
            assert inputs['input_ids'].shape[0] <= self.config.max_seq_len, f"input_ids length exceeds max_seq_len: {inputs['input_ids'].shape[0]} > {self.config.max_seq_len}, may cause errors!!"
        else:
            inputs = self.processor.tokenizer(text=texts, return_tensors="pt", add_special_tokens=False, truncation=True, padding=True, max_length=self.config.max_seq_len)
            pixel_values = self.image_padding['pixel_values']
            image_grid_thw = self.image_padding['image_grid_thw']

        labels = torch.full_like(inputs['input_ids'], -100)
        for batch_idx in range(inputs['input_ids'].size(0)):
            input_ids = inputs['input_ids'][batch_idx]
            assistant_indices = (input_ids == self.assistant_token_id).nonzero(as_tuple=True)[0]

            stop_flag = False
            
            for idx in reversed(assistant_indices):
                if not torch.equal(input_ids[idx:idx+len(self.assistant_prefix_ids)], self.assistant_prefix_ids):
                    continue

                for j in range(idx + len(self.assistant_prefix_ids), len(input_ids)):
                    labels[batch_idx, j] = input_ids[j]
                    if input_ids[j] == self.eos_token_id:
                        stop_flag = True
                        break
                
                if stop_flag:
                    break

        assert inputs['attention_mask'].shape[0] <= self.config.max_seq_len, f"attention_mask length exceeds max_seq_len: {inputs['attention_mask'].shape[0]} > {self.config.max_seq_len}"

        return {
            "input_ids": inputs['input_ids'][:, :self.config.max_seq_len],
            "labels": labels[:, :self.config.max_seq_len],
            "attention_mask": inputs['attention_mask'][:, :self.config.max_seq_len],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw
        }

def save_checkpoint(
    model,
    processor,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    global_step: int,
    is_last: bool = False
) -> None:
    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    
    if accelerator.is_main_process:
        # Manage number of checkpoints
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
            oldest_ckpt = min(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
            shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))

        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, 'tfmr')

        # Save model and tokenizer
        if accelerator.state.deepspeed_plugin.zero_stage != 3:
            model.save_pretrained(output_dir, state_dict=accelerator.get_state_dict(model))
        processor.save_pretrained(output_dir)

        logger.info(f'Model saved in {output_dir}')

    # Handle DeepSpeed stage 3
    if accelerator.state.deepspeed_plugin.zero_stage == 3:
        unwrap_model = accelerator.unwrap_model(model)
        unwrap_model.save_pretrained(
            os.path.join(save_dir, 'tfmr'),
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )

    accelerator.wait_for_everyone()
    logger.info(f'Checkpoint {epoch}-{global_step} saved successfully')

    if args.save_state:
        try:
            accelerator.save_state(save_dir)
        except Exception as e:
            logger.info(f"Failed to save state: {e}")


def train(args: argparse.Namespace) -> None:
    """Main training function"""
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            config=args,
            dir=args.log_dir,
            mode="offline"
        )

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = (
        args.train_bsz_per_gpu * 
        dist.get_world_size() * 
        accelerator.gradient_accumulation_steps
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Freeze components if required
    if args.freeze_vision_tower:
        logger.info("Freezing vision tower parameters")
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.visual.merger.parameters():
            param.requires_grad = True

    if args.freeze_multi_modal_projector:
        logger.info("Freezing multi-modal projector parameters")
        for param in model.visual.merger.parameters():
            param.requires_grad = False

    if args.freeze_language_model:
        logger.info("Freezing language model parameters")
        for param in model.model.parameters():
            param.requires_grad = False

    # Display parameter statistics
    if args.freeze_vision_tower or args.freeze_language_model or args.freeze_multi_modal_projector:
        trainable_params, all_params = 0, 0
        param_status = {}
        for name, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                module_name = name.split('.')[0] if '.' in name else name
                if module_name not in param_status:
                    param_status[module_name] = {'trainable': 0, 'frozen': 0}
                param_status[module_name]['trainable'] += param.numel()
            else:
                module_name = name.split('.')[0] if '.' in name else name
                if module_name not in param_status:
                    param_status[module_name] = {'trainable': 0, 'frozen': 0}
                param_status[module_name]['frozen'] += param.numel()

        accelerator.print(f"Total model params: {all_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M ({trainable_params/all_params*100:.2f}%)")
        accelerator.print("Module-wise parameter status:")
        for module, stats in param_status.items():
            total = stats['trainable'] + stats['frozen']
            if total > 0:
                accelerator.print(f"  - {module}: Total {total/1e6:.2f}M, Trainable {stats['trainable']/1e6:.2f}M ({stats['trainable']/total*100:.2f}%), Frozen {stats['frozen']/1e6:.2f}M ({stats['frozen']/total*100:.2f}%)")

    # Optimizer setup
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = SftDataset(args, processor, accelerator)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4
    )

    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    metric = TrainingMetrics(device=torch.cuda.current_device())
    model.train()

    global_step = 0
    inepoch_step = 0
    start_epoch = 0

    for epoch in range(start_epoch, args.n_epochs):
        train_iter = tqdm(train_dataloader, total=len(train_dataloader)) if accelerator.is_main_process else train_dataloader

        for batch in train_iter:
            if inepoch_step > 0:
                inepoch_step -= 1
                continue

            outputs = model(
                input_ids=batch['input_ids'],
                labels=batch['labels'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                return_dict=True,
                use_cache=False
            )
            loss = outputs.loss
            metric(outputs.logits, batch['labels'], loss)

            accelerator.backward(loss)

            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if (global_step + 1) % (accelerator.gradient_accumulation_steps * 2) == 0:
                    torch.cuda.empty_cache()

            global_step += 1

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                acc, train_loss = metric.get_metric()
                if accelerator.is_main_process:
                    train_iter.set_postfix(
                        epoch=epoch,
                        step=global_step - 1,
                        total_steps=len(train_dataloader),
                        skip=accelerator.optimizer_step_was_skipped,
                        length=len(batch["input_ids"][0]),
                        loss=f"{train_loss:.3f}",
                        acc=f"{acc:.3f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                    wandb.log({
                        'loss': train_loss,
                        'acc': acc,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)

        accelerator.wait_for_everyone()
        save_checkpoint(
            model=model,
            processor=processor, 
            accelerator=accelerator,
            args=args,
            epoch=epoch,
            step=global_step - 1,
            global_step=global_step,
            is_last=True
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretraining Configuration')
    
    # Experiment
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--run_name', type=str, default=None, help='Run name')
    parser.add_argument('--model_path', type=str, default='', help='Pretrained model path')

    # Data
    parser.add_argument('--data_path', type=str, required=True, help='Training data path')
    parser.add_argument('--output_dir', type=str, default='./', help='Model save path')
    parser.add_argument('--max_ckpts', type=int, default=3, help='Max number of checkpoints')
    parser.add_argument('--log_dir', type=str, default='./', help='Log directory')

    # Training
    parser.add_argument('--max_seq_len', type=int, default=4096, help='Max sequence length')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping threshold')
    parser.add_argument('--train_bsz_per_gpu', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0.15, help='Min learning rate ratio')
    parser.add_argument('--warmup_rates', type=float, default=0.05, help='Warmup rate')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--save_steps', type=int, default=2000, help='Checkpoint saving interval')
    parser.add_argument('--save_state', action='store_true', help='Save accelerator state')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume from checkpoint')

    parser.add_argument('--freeze_vision_tower', action='store_true', help='Freeze vision tower')
    parser.add_argument('--freeze_language_model', action='store_true', help='Freeze language model')
    parser.add_argument('--freeze_multi_modal_projector', action='store_true', help='Freeze multi-modal projector')

    # Image
    parser.add_argument('--max_width', type=int, default=420, help='Max image width')
    parser.add_argument('--max_height', type=int, default=360, help='Max image height')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    
    # Set paths
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)
