#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import logging
import argparse
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from cli import HuatuoChatbot  # Assuming this module is available in your environment
from PIL import Image

# --- Configure Logging ---
# Set up the logger to provide clear output information
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VQADataset(Dataset):
    """
    Class for loading VQA datasets.
    Reads data from an XLSX file and matches it with image files.
    """
    def __init__(self, xlsx_path, image_dir, debug_num_samples=None):
        super().__init__()
        try:
            df = pd.read_excel(xlsx_path)
        except FileNotFoundError:
            logger.error(f"Error: XLSX file path '{xlsx_path}' not found")
            raise
        except Exception as e:
            logger.error(f"Error reading XLSX file: {e}")
            raise

        if debug_num_samples and debug_num_samples > 0:
            logger.warning(f"--- Debug mode enabled. Only loading the first {debug_num_samples} records. ---")
            df = df.head(debug_num_samples)

        self.datas = []
        logger.info(f"Loading data from {xlsx_path}...")

        for _, row in df.iterrows():
            question = row['question']
            index_str = str(row['index'])
            image_path = os.path.join(image_dir, f"{index_str}.png")
            
            if not os.path.exists(image_path):
                logger.warning(f"[Skipping] Image file not found: '{image_path}' (Index {index_str})")
                continue

            try:
                # Attempt to open and verify the image file to ensure it is not corrupted
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                logger.error(f"[Skipping] Image file is corrupted or unreadable: '{image_path}' (Index {index_str}). Error: {e}")
                continue

            # Create different prompts for "Direct" and "CoT" modes
            prompt_direct = f"{question}\nPlease directly provide the final answer without any other output."
            prompt_cot = f"{question}\nPlease generate a step by step answer, include all your intermediate reasoning process, and provide the final answer at the end."
            
            self.datas.append({
                'index': index_str,
                'image_path': image_path,
                'query_direct': prompt_direct,
                'query_cot': prompt_cot
            })
        
        logger.info(f"Successfully loaded {len(self.datas)} valid records.")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate_fn function to organize batch data into a dictionary format.
        """
        out_batch = {}
        keys = batch[0].keys()
        for key in keys:
            out_batch[key] = [item[key] for item in batch]
        return out_batch

def run_inference_mode(bot, dataloader, mode):
    """
    Executes the inference process for a specific mode.
    
    Args:
        bot (HuatuoChatbot): The inference bot instance.
        dataloader (DataLoader): Data loader.
        mode (str): Inference mode, 'direct' or 'cot'.

    Returns:
        tuple: A list of results and the total inference time.
    """
    results = []
    total_time = 0.0
    query_key = f'query_{mode}'

    # Use torch.no_grad() to disable gradient calculation, saving memory and speeding up inference
    with torch.no_grad():
        # Use tqdm to create a progress bar
        pbar = tqdm(dataloader, desc=f"Inference ({mode.upper()} Mode)")
        
        for batch in pbar:
            # Current implementation assumes batch_size=1, so it loops once
            for i in range(len(batch['index'])):
                index = batch['index'][i]
                image_path = batch['image_path'][i]
                
                # [Critical Section] Wrap single sample inference in a try-except block
                try:
                    logger.info(f"-> Processing Index: {index} ({mode.upper()} Mode)")

                    # Execute inference
                    query = batch[query_key][i]
                    start_time = time.time()
                    response = bot.inference(query, [image_path])
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    results.append({"index": index, "prediction": response[0]})
                    
                    logger.info(f"<- Successfully processed Index: {index} ({mode.upper()}: {inference_time:.2f}s)")

                except Exception as e:
                    # If any error occurs during inference, log the error and skip the sample
                    logger.error(f"!!!!!!!! Critical error processing Index {index} ({mode.upper()} Mode). Skipped. Error: {e} !!!!!!!!")
                    
                    # Record the error in results for later analysis
                    error_msg = f"SKIPPED due to error: {e}"
                    results.append({"index": index, "prediction": error_msg})
                    
                    # Continue to the next sample
                    continue
    
    # Sort results by index to ensure output order matches input order
    results.sort(key=lambda x: int(x['index']))
    
    return results, total_time

def run_inference(bot, dataloader, modes):
    """
    Executes the core inference process.
    
    Args:
        bot (HuatuoChatbot): The inference bot instance.
        dataloader (DataLoader): Data loader.
        modes (list): List of modes to run, e.g., ['direct', 'cot'].

    Returns:
        dict: A dictionary containing results and inference times for each mode.
    """
    results = {}
    
    for mode in modes:
        logger.info(f"\nStarting {mode.upper()} mode inference...")
        mode_results, mode_time = run_inference_mode(bot, dataloader, mode)
        results[mode] = {
            'results': mode_results,
            'time': mode_time
        }
        logger.info(f"{mode.upper()} mode inference completed. Time elapsed: {mode_time:.2f}s")
    
    return results

def load_existing_timing(timing_file):
    """
    Loads an existing timing file.
    
    Args:
        timing_file (str): Path to the timing file.
        
    Returns:
        dict: Existing timing data, or an empty structure if the file does not exist.
    """
    if os.path.exists(timing_file):
        try:
            with open(timing_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to read existing timing file: {e}. Creating a new file.")
    
    # Return empty timing data structure
    return {
        "model_name": "",
        "modes": {},
        "summary": {},
        "last_updated": ""
    }

def save_timing_results(model_name, mode_times, output_dir):
    """
    Saves timing results to a JSON file, supporting incremental updates.
    
    Args:
        model_name (str): Name of the model.
        mode_times (dict): Inference times for each mode, format: {'direct': time, 'cot': time}.
        output_dir (str): Output directory.
    """
    timing_file = os.path.join(output_dir, "timing_results", "HuatuoGPT-Vision.json")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Load existing data
    timing_data = load_existing_timing(timing_file)
    
    # Update model name
    if not timing_data["model_name"]:
        timing_data["model_name"] = model_name
    elif timing_data["model_name"] != model_name:
        logger.warning(f"Model name mismatch! Existing: {timing_data['model_name']}, Current: {model_name}")
    
    # Update timing data for each mode
    for mode, time_seconds in mode_times.items():
        if time_seconds > 0:  # Only update if the mode actually ran
            timing_data["modes"][mode] = {
                "dataset_total_time_seconds": round(time_seconds, 2),
                "dataset_total_time_minutes": round(time_seconds / 60, 2),
                "dataset_total_time_hours": round(time_seconds / 3600, 2),
                "updated_at": current_time
            }
            logger.info(f"Updated {mode.upper()} mode timing: {time_seconds:.2f}s")
    
    # Calculate summary information
    total_time_seconds = sum(
        timing_data["modes"].get(mode, {}).get("dataset_total_time_seconds", 0)
        for mode in ["direct", "cot"]
    )
    
    modes_count = len([mode for mode in ["direct", "cot"] if mode in timing_data["modes"]])
    
    # Calculate time ratio
    direct_time = timing_data["modes"].get("direct", {}).get("dataset_total_time_seconds", 0)
    cot_time = timing_data["modes"].get("cot", {}).get("dataset_total_time_seconds", 0)
    time_ratio = round(cot_time / direct_time, 2) if direct_time > 0 else 0
    
    timing_data["summary"] = {
        "total_time_seconds": round(total_time_seconds, 2),
        "total_time_minutes": round(total_time_seconds / 60, 2),
        "total_time_hours": round(total_time_seconds / 3600, 2),
        "modes_count": modes_count,
        "updated_at": current_time,
        "time_ratio": {
            "description": "cot/direct",
            "ratio": time_ratio
        }
    }
    
    timing_data["last_updated"] = current_time
    
    os.makedirs(os.path.dirname(timing_file), exist_ok=True)
    # Save updated data
    with open(timing_file, 'w', encoding='utf-8') as f:
        json.dump(timing_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Timing results saved to: {timing_file}")
    logger.info(f"Total current inference time: {total_time_seconds:.2f}s ({total_time_seconds/60:.2f} min)")
    if modes_count == 2:
        logger.info(f"CoT/Direct Time Ratio: {time_ratio}")

def main(args):
    """
    Main execution function.
    """
    # Set random seed to ensure reproducibility
    set_seed(args.seed)
    
    # Check and create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory '{args.output_dir}' is ready.")

    # Initialize model
    logger.info(f"Initializing model: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    bot = HuatuoChatbot(args.model_path) # HuatuoChatbot should handle device assignment internally
    bot.gen_kwargs['max_new_tokens'] = args.max_new_tokens
    logger.info("Model loaded.")

    # Get model name
    model_name = os.path.basename(args.model_path)

    # Load dataset
    dataset = VQADataset(args.data_path, args.image_dir, args.debug_num_samples)
    original_data_count = len(dataset)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=VQADataset.collate_fn)

    # Determine modes to run
    modes_to_run = []
    if args.run_direct:
        modes_to_run.append('direct')
    if args.run_cot:
        modes_to_run.append('cot')
    
    if not modes_to_run:
        logger.error("Please specify at least one running mode (--run_direct or --run_cot)")
        return

    # Start Inference
    logger.info("\n" + "="*50)
    logger.info(f"  Starting inference, Modes: {', '.join([m.upper() for m in modes_to_run])}")
    logger.info("="*50)
    
    inference_results = run_inference(bot, dataloader, modes_to_run)
    
    # --- Save Results ---
    mode_times = {}
    
    for mode in modes_to_run:
        results = inference_results[mode]['results']
        mode_time = inference_results[mode]['time']
        mode_times[mode] = mode_time
        
        if results:
            if len(results) != original_data_count:
                logger.warning(f"[Warning] '{mode.upper()}' mode final result count ({len(results)}) does not match original data count ({original_data_count})! This may be due to skipping some error samples.")
            
            # Use os.path.join to construct full output path
            output_file = os.path.join(args.output_dir, "HuatuoGPT-Vision", f"HuatuoGPT-Vision_{mode}.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results for '{mode.upper()}' mode saved to: {output_file}")

    # Save timing results (incremental update)
    if any(results for results in [inference_results[mode]['results'] for mode in modes_to_run]):
        save_timing_results(model_name, mode_times, args.output_dir)

    if not any(inference_results[mode]['results'] for mode in modes_to_run):
         logger.error("\nNo valid results generated during inference.")

    logger.info("\nAll inference tasks completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference using HuatuoGPT-Vision and generate two types of responses (Supports individual run modes)')
    
    # --- Input Arguments ---
    parser.add_argument('--data_path', default='./dataset/M3CoTBench.xlsx"', type=str, help='Path to the input XLSX file.')
    parser.add_argument('--image_dir', default='./dataset/images', type=str, help='Directory path containing images.')
    parser.add_argument('--model_path', default='./pretrain/HuatuoGPT-Vision-7B', type=str, help='Path to the pretrained model.')
    
    # --- Output Arguments ---
    parser.add_argument('--output_dir', default='./final_output', type=str, help='Directory to store output JSON files.')

    # --- Inference Configuration Arguments ---
    parser.add_argument('--max_new_tokens', default=2048, type=int, help='Maximum number of tokens to generate. CoT mode may require more tokens.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference. Strongly recommended to keep as 1.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    
    # --- Mode Selection Arguments ---
    parser.add_argument('--run_direct', action='store_true', help='Run Direct mode inference')
    parser.add_argument('--run_cot', action='store_true', help='Run CoT mode inference')
    
    # --- Debugging Arguments ---
    parser.add_argument('--debug_num_samples', default=0, type=int, help='Load only a specific number of samples for debugging. Set to 0 or leave unset to load all.')
    
    args = parser.parse_args()
    
    # If no mode is specified, default to running both modes
    if not args.run_direct and not args.run_cot:
        args.run_direct = True
        args.run_cot = True
        logger.info("No run mode specified, running all modes (Direct + CoT)")
    
    if args.batch_size != 1:
        logger.warning("Warning: Current implementation is best suited for batch_size=1. Using other values may lead to unexpected behavior.")
        
    main(args)