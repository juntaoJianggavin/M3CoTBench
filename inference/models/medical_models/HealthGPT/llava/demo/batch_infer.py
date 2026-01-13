#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
# 确保可以找到llava库
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import json
from PIL import Image
import torch
import transformers
import tokenizers
import pandas as pd
from tqdm import tqdm
import time  # 新增：用于计时
from datetime import datetime # 新增：用于记录更新时间

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM
from utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square
from utils import com_vision_args # 假设 com_vision_args 在 utils 中定义
from llava.peft import LoraConfig, get_peft_model

def update_timing_report(report_path, model_name, mode, elapsed_time_seconds):
    """
    读取、更新并保存计时报告JSON文件。支持增量更新。

    Args:
        report_path (str): 计时报告JSON文件的完整路径。
        model_name (str): 正在评估的模型名称。
        mode (str): 当前的评估模式 ('cot' 或 'direct')。
        elapsed_time_seconds (float): 本次模式运行的总耗时（秒）。
    """
    # 确保报告所在的目录存在
    report_dir = os.path.dirname(report_path)
    os.makedirs(report_dir, exist_ok=True)

    # 初始化或加载现有数据
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # 如果文件损坏或为空，则重新初始化
            data = {
                "model_name": model_name,
                "modes": {},
                "summary": {}
            }
    else:
        data = {
            "model_name": model_name,
            "modes": {},
            "summary": {}
        }
    
    # 更新当前模式的计时信息
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['last_updated'] = now_str
    data['model_name'] = model_name # 确保模型名称是最新的
    
    data['modes'][mode] = {
        "dataset_total_time_seconds": round(elapsed_time_seconds, 2),
        "dataset_total_time_minutes": round(elapsed_time_seconds / 60, 2),
        "dataset_total_time_hours": round(elapsed_time_seconds / 3600, 2),
        "updated_at": now_str
    }

    # 重新计算并更新Summary部分
    summary = {}
    all_modes = data.get('modes', {})
    total_seconds = sum(info.get('dataset_total_time_seconds', 0) for info in all_modes.values())
    
    summary['total_time_seconds'] = round(total_seconds, 2)
    summary['total_time_minutes'] = round(total_seconds / 60, 2)
    summary['total_time_hours'] = round(total_seconds / 3600, 2)
    summary['modes_count'] = len(all_modes)
    summary['updated_at'] = now_str

    # 计算时间比率 (cot/direct)
    cot_time = all_modes.get('cot', {}).get('dataset_total_time_seconds')
    direct_time = all_modes.get('direct', {}).get('dataset_total_time_seconds')

    if cot_time is not None and direct_time is not None and direct_time > 0:
        summary['time_ratio'] = {
            "description": "cot/direct",
            "ratio": round(cot_time / direct_time, 2)
        }
    else:
        summary['time_ratio'] = {
            "description": "cot/direct",
            "ratio": None # 当其中一个模式的数据不存在时，比率无法计算
        }

    data['summary'] = summary

    # 将更新后的数据写回JSON文件
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Timing report updated successfully at: {report_path}")


def batch_infer():
    """
    主函数，用于批量推理VQA数据集。
    """
    parser = argparse.ArgumentParser(description="Batch inference script for HealthGPT model.")
    
    # --- 模型和路径参数 ---
    parser.add_argument('--model_name_or_path', type=str, default='/z_data/byl/MME-CoT-benchmarks/Med-CoT-Eval_final/pretrain/HealthGPT/pretrain/Phi-3-mini-4k-instruct')
    parser.add_argument('--vit_path', type=str, default='/z_data/byl/MME-CoT-benchmarks/Med-CoT-Eval_final/pretrain/HealthGPT/pretrain/clip-vit-large-patch14-336')
    parser.add_argument('--hlora_path', type=str, default='/z_data/byl/MME-CoT-benchmarks/Med-CoT-Eval_final/pretrain/HealthGPT/pretrain/HealthGPT-M3', help="Path to the hlora weights file (com_hlora_weights.bin).")

    # --- 模型配置参数 ---
    parser.add_argument('--dtype', type=str, default='FP32', choices=['FP16', 'FP32', 'BF16'], help="Computation data type.")
    parser.add_argument('--attn_implementation', type=str, default=None, help="Attention implementation (e.g., 'flash_attention_2').")
    parser.add_argument('--hlora_r', type=int, default=64)
    parser.add_argument('--hlora_alpha', type=int, default=128)
    parser.add_argument('--hlora_dropout', type=float, default=0.0)
    parser.add_argument('--hlora_nums', type=int, default=4)
    parser.add_argument('--vq_idx_nums', type=int, default=8192)
    parser.add_argument('--instruct_template', type=str, default='phi3_instruct')

    # --- 数据集和评估参数 ---
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input XLSX file.")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output JSON file for predictions.")
    parser.add_argument('--eval_mode', type=str, required=True, choices=['direct', 'cot'], help="Evaluation mode: 'direct' for direct answers, 'cot' for Chain-of-Thought.")

    # --- 推理生成参数 ---
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling.")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    
    args = parser.parse_args()

    # --- 1. 模型加载和初始化 (这部分不计时) ---
    print("Initializing model and tokenizer...")
    model_dtype = {'FP32': torch.float32, 'FP16': torch.float16, 'BF16': torch.bfloat16}[args.dtype]

    model = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=model_dtype
    )

    lora_config = LoraConfig(
        r=args.hlora_r,
        lora_alpha=args.hlora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=args.hlora_dropout,
        bias='none',
        task_type="CAUSAL_LM",
        lora_nums=args.hlora_nums,
    )
    model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model, args.vq_idx_nums)
    print(f"Number of new tokens added: {num_new_tokens}")

    com_vision_args.model_name_or_path = args.model_name_or_path
    com_vision_args.vision_tower = args.vit_path
    com_vision_args.version = args.instruct_template

    model.get_model().initialize_vision_modules(model_args=com_vision_args)
    model.get_vision_tower().to(dtype=model_dtype)

    print(f"Loading weights from {args.hlora_path}...")
    model = load_weights(model, args.hlora_path)
    model.eval()
    model.to(model_dtype).cuda()
    print("Model loaded successfully and moved to GPU.")

    # --- 2. 数据加载和准备 (这部分不计时) ---
    print(f"Loading data from {args.data_path}...")
    df = pd.read_excel(args.data_path)
    results = []

    if args.eval_mode == 'direct':
        prompt_suffix = "\nPlease directly provide the final answer without any other output."
    elif args.eval_mode == 'cot':
        prompt_suffix = "\nPlease generate a step by step answer, include all your intermediate reasoning process, and provide the final answer at the end."
    else:
        raise ValueError("Invalid eval_mode. Choose 'direct' or 'cot'.")

    # --- 3. 批量推理循环 (这部分需要计时) ---
    print(f"Starting batch inference in '{args.eval_mode}' mode...")
    
    # 新增：开始计时
    inference_start_time = time.perf_counter()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        idx = row['index']
        question = row['question']
        
        # 新增：处理 question 的类型问题
        if pd.isna(question):
            print(f"Warning: Question is NaN for index {idx}. Skipping.")
            continue
        
        # 确保 question 是字符串类型
        question = str(question).strip()
        
        # 如果问题为空字符串，跳过
        if not question:
            print(f"Warning: Question is empty for index {idx}. Skipping.")
            continue
        
        img_path = os.path.join(args.image_dir, f"{idx}.png")
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for index {idx} at {img_path}. Skipping.")
            continue

        full_question = question + prompt_suffix
            
        qs = DEFAULT_IMAGE_TOKEN + '\n' + full_question
        conv = conversation_lib.conv_templates[args.instruct_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze(0)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = expand2square(image, tuple(int(x*255) for x in model.get_vision_tower().image_processor.image_mean))
            image_tensor = model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0)
            image_size = image.size
        except Exception as e:
            print(f"Error processing image for index {idx}: {e}. Skipping.")
            continue

        with torch.inference_mode():
            output_ids = model.base_model.model.generate(
                input_ids,
                images=image_tensor.to(dtype=model_dtype, device='cuda', non_blocking=True),
                image_sizes=[image_size],
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        
        try:
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)[:-8]
        except IndexError:
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        results.append({
            "index": str(idx),
            "prediction": response.strip()
        })

    # 新增：结束计时并计算总时间
    inference_end_time = time.perf_counter()
    total_inference_time = inference_end_time - inference_start_time
    print(f"Total inference time for '{args.eval_mode}' mode: {total_inference_time:.2f} seconds.")

    # --- 4. 保存结果和计时报告 ---
    # 保存预测结果
    print(f"Batch inference complete. Saving prediction results to {args.output_path}...")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 新增：更新并保存计时报告
    # 从模型路径中提取一个简洁的模型名
    model_name_simple = os.path.basename(args.model_name_or_path)
    # 定义计时报告的路径
    output_dir = os.path.dirname(args.output_path)
    timing_report_path = os.path.join(output_dir, 'timing_results', 'HealthGPT.json')
    
    update_timing_report(
        report_path=timing_report_path,
        model_name=model_name_simple, # 使用从路径中提取的简洁名称
        mode=args.eval_mode,
        elapsed_time_seconds=total_inference_time
    )

    print("Done.")


if __name__ == "__main__":
    batch_infer()