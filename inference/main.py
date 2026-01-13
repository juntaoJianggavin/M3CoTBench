import os
import sys
# Ensure project root is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import json
import time
import pickle
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from config.config import Config
from data.dataset import MedicalVQADataset
from inference.engine import InferenceEngine
from models.model_registry import list_available_models
from utils.logger import setup_logger
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import Subset
from datetime import timedelta

import torch.distributed as dist

# ==============================================================================
# Helper Functions
# ==============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Medical VQA Evaluation (V8 - Persistent Summary)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Name of a single model to evaluate. If not provided, evaluates all models in the config.")
    parser.add_argument("--data_path", type=str, default="../inferencedataset/M3CoTBench.xlsx")
    parser.add_argument("--image_dir", type=str, default="../inferenced/dataset/images")
    parser.add_argument("--output_dir", type=str, default="final_output")
    parser.add_argument("--mode", type=str, choices=["cot", "direct", "both"], default="cot")
    parser.add_argument("--list_models", action="store_true", help="List all available models and exit.")
    parser.add_argument("--pre_prompt", type=str, default="")
    parser.add_argument("--post_prompt", type=str, default="")
    parser.add_argument("--max_image_size", type=int, default=768, help="Maximum size for input images.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N samples for quick testing.")
    parser.add_argument("--timeout", type=int, default=7200, help="Distributed communication timeout in seconds, default is 30 minutes.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from existing checkpoints, skipping completed shards.")
    parser.add_argument("--specific_indices", type=str, default=None, 
                        help="Process only specific sample indices, comma-separated, e.g., '0,5,10' or a single index '99'")
    return parser.parse_args()


def get_checkpoint_path(output_dir, model_name, mode, shard_index):
    """Get checkpoint file path."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    checkpoint_file = f"{safe_model_name}_{mode}_shard{shard_index}.checkpoint"
    return os.path.join(checkpoint_dir, checkpoint_file)


def load_checkpoint(checkpoint_path, logger):
    """Load checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        logger.debug(f"Checkpoint file does not exist: {checkpoint_path}")
        return None
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        # Compatible with new and old formats
        num_results = len(checkpoint.get('summary', {}).get('results', checkpoint.get('results', [])))
        logger.info(f"Successfully loaded checkpoint: {checkpoint_path} (containing {num_results} results)")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def save_checkpoint(checkpoint_path, checkpoint_data, logger):
    """Safely save checkpoint data to file."""
    try:
        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        os.rename(temp_path, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def prepare_dataset(args, mode, num_shards, shard_index, logger):
    """Prepare the dataset and return the correct data shard based on current process info (lazy-loaded Dataset object)."""
    dataset = MedicalVQADataset(
        data_path=args.data_path, image_dir=args.image_dir, mode=mode,
        pre_prompt=args.pre_prompt, post_prompt=args.post_prompt,
        max_image_size=args.max_image_size
    )
    # If specific indices are specified, process only these indices
    if args.specific_indices is not None:
        indices = [int(x.strip()) for x in args.specific_indices.split(',')]
        dataset = Subset(dataset, indices)
        logger.info(f"Processing only specified indices: {indices}")
        return dataset
    if args.limit is not None and args.limit > 0:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    
    if num_shards > 1:
        total = len(dataset)
        per_shard = (total + num_shards - 1) // num_shards
        start = shard_index * per_shard
        end = min((shard_index + 1) * per_shard, total)
        if start < total:
            dataset_shard = Subset(dataset, range(start, end))
            logger.info(f"Sharding: Process {shard_index}/{num_shards}, processing sample indices {start}~{end-1}, total {len(dataset_shard)} items")
            return dataset_shard
        else:
            logger.info(f"Sharding: Process {shard_index}/{num_shards}, no data to process")
            return Subset(dataset, []) # Return an empty dataset subset
    return dataset


def run_and_report_status_for_shard(engine, dataset, output_dir, dataset_mode, shard_index, 
                                    model_name, resume_from_checkpoint, logger) -> Tuple[bool, Dict[str, Any]]:
    """
    Run inference for a single shard and return success status along with detailed timing statistics.
    """
    checkpoint_path = get_checkpoint_path(output_dir, model_name, dataset_mode, shard_index)
    
    default_stats = {
        "total_successful_time_s": 0.0, "total_wasted_time_s": 0.0,
        "successful_item_count": 0, "failed_item_count": 0, "total_item_count": 0
    }

    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        logger.info(f"Found existing checkpoint {checkpoint_path} and resume enabled, treating this shard task as successful.")
        checkpoint = load_checkpoint(checkpoint_path, logger)
        stats = checkpoint.get('summary', {}).get('stats', default_stats)
        return True, stats

    logger.info(f"Starting inference, {len(dataset)} samples to process")
    
    try:
        # engine.run now returns a dictionary containing results and stats
        summary = engine.run(dataset=dataset)
        
        # Save the complete summary (results + stats) to checkpoint
        save_checkpoint(checkpoint_path, {'summary': summary, 'timestamp': time.time()}, logger)
        
        stats = summary.get('stats', default_stats)
        logger.info(f"Shard {shard_index} completed. Success: {stats['successful_item_count']}, Failed: {stats['failed_item_count']}.")
        logger.info(f"  -> Success time: {stats['total_successful_time_s']:.2f}s, Failed attempt time: {stats['total_wasted_time_s']:.2f}s")
        
        return True, stats

    except Exception as e:
        logger.error(f"Severe error in shard {shard_index} during inference or checkpoint saving: {e}", exc_info=True)
        return False, default_stats


def merge_results_and_stats(output_dir, model_name, mode, num_shards, logger) -> Dict[str, Any]:
    """Read results and stats from all shard checkpoint files, merge, save, and then clean up."""
    logger.info(f"Main process starting to merge results from disk for model '{model_name}' mode '{mode}'...")
    all_results = []
    
    aggregated_stats = {
        "total_successful_time_s": 0.0, "total_wasted_time_s": 0.0,
        "successful_item_count": 0, "failed_item_count": 0, "total_item_count": 0
    }

    for i in range(num_shards):
        checkpoint_path = get_checkpoint_path(output_dir, model_name, mode, i)
        checkpoint = load_checkpoint(checkpoint_path, logger)
        if checkpoint and 'summary' in checkpoint:
            summary = checkpoint['summary']
            shard_results = summary.get('results', [])
            shard_stats = summary.get('stats', {})
            
            all_results.extend(shard_results)
            
            for key in aggregated_stats:
                aggregated_stats[key] += shard_stats.get(key, 0)
            
            logger.info(f"  Shard {i}: Loaded {len(shard_results)} results. Current total: {len(all_results)}")
        else:
            logger.warning(f"  Warning: Failed to load results from checkpoint of shard {i}.")
    
    if not all_results:
        logger.warning("No results collected, skipping saving final file.")
        return aggregated_stats
    
    logger.info(f"All shard results merged, total {len(all_results)} items. Sorting and saving now...")
    try:
        all_results.sort(key=lambda x: int(x.get('index', 0)))
        safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        model_output_dir = os.path.join(output_dir, safe_model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        final_filepath = os.path.join(model_output_dir, f"{safe_model_name}_{mode}.json")
        with open(final_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Final results successfully saved to: {final_filepath}")
        
        logger.info("Starting cleanup of temporary checkpoint files...")
        for i in range(num_shards):
            checkpoint_path = get_checkpoint_path(output_dir, model_name, mode, i)
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except OSError as e:
                    logger.error(f"Failed to delete checkpoint {checkpoint_path}: {e}")
        logger.info("Temporary checkpoint file cleanup completed.")
    except Exception as e:
        logger.error(f"Error saving final file or cleaning checkpoints: {e}", exc_info=True)
        logger.error("Error occurred, temporary checkpoint files will not be deleted for safety.")
    
    return aggregated_stats


def save_or_update_summary(output_dir: str, model_name: str, mode: str, new_stats: Dict[str, Any], logger):
    """
    Read, update, and save a JSON file containing statistical summaries for all modes.
    This file is unique to each model and accumulates run results from different modes.
    """
    safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    model_output_dir = os.path.join(output_dir, safe_model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    summary_filepath = os.path.join(model_output_dir, f"{safe_model_name}_summary.json")
    
    logger.info(f"Updating statistical summary file: {summary_filepath}")
    
    all_summaries = {}
    try:
        if os.path.exists(summary_filepath):
            with open(summary_filepath, 'r', encoding='utf-8') as f:
                all_summaries = json.load(f)
                logger.debug("Loaded existing summary file.")
    except json.JSONDecodeError:
        logger.warning(f"Summary file {summary_filepath} exists but format is incorrect, creating a new one.")
        all_summaries = {}
    except Exception as e:
        logger.error(f"Error reading summary file {summary_filepath}: {e}. Will attempt to create a new one.")
        all_summaries = {}

    all_summaries[mode] = new_stats
    logger.debug(f"Summary content updated, data for mode '{mode}' added/overwritten.")

    try:
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=4, ensure_ascii=False)
        logger.info(f"Statistical summary successfully updated and saved to: {summary_filepath}")
    except Exception as e:
        logger.error(f"Failed to save updated summary file: {e}", exc_info=True)

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    args = parse_args()
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    logger = setup_logger()
    
    logger.info("Starting Medical VQA Evaluation Program (V8 - Persistent Summary)")
    logger.info(f"Distributed info: Num processes={accelerator.num_processes}, Current process={accelerator.process_index}")
    
    config = Config(args.config) if args.config else Config()

    if args.list_models:
        if accelerator.is_main_process:
            print("Available models:", list_available_models(config))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    models_to_evaluate = [args.model] if args.model else config.get_all_model_names()
    modes_to_run = ["cot", "direct"] if args.mode == "both" else [args.mode]
    
    num_shards = accelerator.num_processes
    shard_index = accelerator.process_index
    
    try:
        for model_name in models_to_evaluate:
            logger.info(f"========== Starting evaluation for model: {model_name} ==========")
            engine = InferenceEngine(model_name=model_name, config=config, accelerator=accelerator)
            
            try:
                for mode in modes_to_run:
                    try:
                        logger.info(f"--- Starting evaluation mode: {mode} ---")
                        dataset_shard = prepare_dataset(args, mode, num_shards, shard_index, logger)
                        
                        inference_successful = True
                        
                        if len(dataset_shard) > 0:
                            inference_successful, _ = run_and_report_status_for_shard(
                                engine=engine, dataset=dataset_shard, output_dir=args.output_dir,
                                dataset_mode=mode, shard_index=shard_index,
                                model_name=model_name, resume_from_checkpoint=args.resume_from_checkpoint, 
                                logger=logger
                            )
                        else:
                            logger.info("No data in current shard, skipping inference.")
                        
                        local_success_tensor = torch.tensor([1.0 if inference_successful else 0.0], device=accelerator.device)
                        dist.all_reduce(local_success_tensor, op=dist.ReduceOp.MIN)

                        if local_success_tensor.item() == 0.0:
                            raise RuntimeError(f"Mode '{mode}' failed on at least one process. Check logs for details.")
                        
                        accelerator.wait_for_everyone()
                        
                        if accelerator.is_main_process:
                            logger.info(f"All shards completed successfully. Main process starting to merge results and stats...")
                            final_stats = merge_results_and_stats(args.output_dir, model_name, mode, num_shards, logger)
                            
                            logger.info("="*70 + f"\nFinal Statistical Summary for Mode '{mode}'\n" + "="*70)
                            total_items = final_stats.get('total_item_count', 0)
                            success_items = final_stats.get('successful_item_count', 0)
                            failed_items = final_stats.get('failed_item_count', 0)
                            success_rate = (success_items / total_items * 100) if total_items > 0 else 0
                            avg_success_time = (final_stats.get('total_successful_time_s', 0) / success_items) if success_items > 0 else 0
                            
                            logger.info(f"  - Data Integrity: {success_items + failed_items} / {total_items} (Should be 100%)")
                            logger.info(f"  - Success Rate: {success_rate:.2f}% ({success_items}/{total_items})")
                            logger.info(f"    - Successful Samples: {success_items}")
                            logger.info(f"    - Failed Samples: {failed_items}")
                            logger.info(f"  - Total Time for Successful Samples: {final_stats.get('total_successful_time_s', 0):.2f} s")
                            logger.info(f"  - Average Time per Successful Sample: {avg_success_time:.2f} s")
                            logger.info(f"  - Total Time for Failed Attempts: {final_stats.get('total_wasted_time_s', 0):.2f} s")
                            logger.info("="*70)

                            # Save or update summary file
                            save_or_update_summary(
                                output_dir=args.output_dir,
                                model_name=model_name,
                                mode=mode,
                                new_stats=final_stats,
                                logger=logger
                            )

                        accelerator.wait_for_everyone()

                    except Exception as mode_error:
                        if accelerator.is_main_process:
                            logger.error(f"Error occurred while processing mode '{mode}', skipping this mode. Error: {mode_error}")
                        accelerator.wait_for_everyone()
                        continue

            finally:
                engine.cleanup()
                logger.info(f"Engine cleanup completed for model {model_name}")

    except Exception as e:
        logger.error(f"Unrecoverable fatal error in main program: {e}", exc_info=True)
    
    finally:
        logger.info(f"Process {shard_index} has completed all tasks, preparing to exit.")
        if dist.is_initialized():
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.info("Main process shutting down distributed environment...")
            dist.destroy_process_group()
            logger.info(f"Distributed environment for process {shard_index} shut down.")

if __name__ == "__main__":
    main()