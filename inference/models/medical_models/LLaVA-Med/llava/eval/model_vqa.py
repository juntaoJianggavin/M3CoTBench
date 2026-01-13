import argparse
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
import time
from datetime import datetime

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n chunks evenly"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    """Get the k-th chunk of the list"""
    chunks = split_list(lst, n)
    return chunks[k]

def update_timing_json(timing_file, model_name, mode, total_time_seconds):
    """Update timing JSON file, supporting incremental updates"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total_time_minutes = round(total_time_seconds / 60, 2)
    total_time_hours = round(total_time_seconds / 3600, 2)
    
    new_timing_data = {
        "dataset_total_time_seconds": round(total_time_seconds, 2),
        "dataset_total_time_minutes": total_time_minutes,
        "dataset_total_time_hours": total_time_hours,
        "updated_at": current_time
    }
    
    if os.path.exists(timing_file):
        try:
            with open(timing_file, 'r', encoding='utf-8') as f:
                timing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            timing_data = {}
    else:
        timing_data = {}
    
    if "model_name" not in timing_data:
        timing_data["model_name"] = model_name
    if "modes" not in timing_data:
        timing_data["modes"] = {}
    
    timing_data["modes"][mode] = new_timing_data
    
    os.makedirs(os.path.dirname(timing_file), exist_ok=True)
    
    with open(timing_file, 'w', encoding='utf-8') as f:
        json.dump(timing_data, f, ensure_ascii=False, indent=2)
    
    print(f"Timing data updated for mode '{mode}': {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes)")

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    df = pd.read_excel(os.path.expanduser(args.question_file))
    questions = []
    for _, row in df.iterrows():
        questions.append({
            "index": str(row["index"]),
            "question": row["question"]
        })

    # ==================== New debugging logic starts ====================
    # If the number of debug samples is set, truncate the list for debugging
    if args.debug_samples is not None and args.debug_samples > 0:
        print(f"Debug mode enabled: Using the first {args.debug_samples} samples.")
        questions = questions[:args.debug_samples]
    # ==================== New debugging logic ends ====================

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    results = []

    system_prompt = (
        "You are a vision-language assistant capable of analyzing medical images and answering related questions. "
        "Please provide accurate and helpful answers based on the image content."
    )

    cot_prompt = "\nPlease generate a step by step answer, include all your intermediate reasoning process, and provide the final answer at the end."
    direct_prompt = "\nPlease directly provide the final answer without any other output."

    inference_start_time = time.time()
    
    for line in tqdm(questions):
        idx = line["index"]
        raw_question = line["question"]

        if args.mode == "cot":
            mode_prompt = cot_prompt
        else:
            mode_prompt = direct_prompt

        cur_prompt = f"{system_prompt}\n{raw_question}{mode_prompt}"

        image_file = f"{idx}.jpg"
        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(image_path):
            image_file = f"{idx}.png"
            image_path = os.path.join(args.image_folder, image_file)
            if not os.path.exists(image_path):
                print(f"Warning: image file for index {idx} not found, skipping.")
                results.append({
                    "index": idx,
                    "prediction": "Image not found"
                })
                continue

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + cur_prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + cur_prompt

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        results.append({
            "index": idx,
            "prediction": outputs
        })

    inference_end_time = time.time()
    total_inference_time = inference_end_time - inference_start_time

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    timing_dir = os.path.join(os.path.dirname(os.path.dirname(answers_file)), "timing_results")
    os.makedirs(os.path.dirname(timing_dir), exist_ok=True)
    timing_file = os.path.join(timing_dir,"LLaVA-Med.json")
    
    update_timing_json(timing_file, model_name, args.mode, total_inference_time)
    
    print(f"Results saved to: {answers_file}")
    print(f"Timing results saved to: {timing_file}")
    print(f"Total inference time for {args.mode} mode: {total_inference_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./pretrained/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./dataset/images")
    parser.add_argument("--question-file", type=str, default="./dataset/M3CoTBench.xlsx")
    parser.add_argument("--answers-file", type=str, default="./final_output/LLaVA-Med/LLaVA-Med.json")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--mode", type=str, choices=["direct", "cot"], default="direct", help="direct or cot mode for prompt")
    
    # ==================== New command line arguments ====================
    parser.add_argument("--debug-samples", type=int, default=None, help="Number of samples to use for debugging. If not provided, all samples are processed.")
    # ====================================================================

    args = parser.parse_args()

    if args.mode == "cot":
        args.answers_file = args.answers_file.replace(".json", "_cot.json")
    else:
        args.answers_file = args.answers_file.replace(".json", "_direct.json")
    
    eval_model(args)