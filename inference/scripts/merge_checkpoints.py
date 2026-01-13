import os
import sys
import argparse
import json
import pickle
import logging
from typing import List, Dict, Any

# ==============================================================================
# Logger Setup
# ==============================================================================
def setup_logger():
    """Configures a simple logger to output messages to the console."""
    logger = logging.getLogger("Merger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# ==============================================================================
# Core Helper Functions (Extracted and modified from your original script)
# ==============================================================================

def get_checkpoint_path(output_dir: str, model_name: str, mode: str, shard_index: int) -> str:
    """Get the checkpoint file path (Exactly consistent with your original script)."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    checkpoint_file = f"{safe_model_name}_{mode}_shard{shard_index}.checkpoint"
    return os.path.join(checkpoint_dir, checkpoint_file)

def load_checkpoint(checkpoint_path: str, logger: logging.Logger) -> Any:
    """Load checkpoint from file (Exactly consistent with your original script)."""
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
        logger.error(f"Failed to load checkpoint: {checkpoint_path}, Error: {e}")
        return None

def save_or_update_summary(output_dir: str, model_name: str, mode: str, new_stats: Dict[str, Any], logger: logging.Logger):
    """
    Read, update, and save a JSON file containing statistical summaries for all modes (Exactly consistent with your original script).
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

def merge_and_finalize(model_name: str, mode: str, output_dir: str, num_shards: int, cleanup: bool, logger: logging.Logger):
    """
    Read results and stats from all shard checkpoint files, merge, save, and then selectively clean up.
    """
    logger.info(f"Starting to merge results for model '{model_name}' mode '{mode}'...")
    all_results = []
    
    aggregated_stats = {
        "total_successful_time_s": 0.0, "total_wasted_time_s": 0.0,
        "successful_item_count": 0, "failed_item_count": 0, "total_item_count": 0
    }

    # 1. Load and aggregate data from all shards
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
            logger.warning(f"  Warning: Failed to load results from checkpoint '{checkpoint_path}' of shard {i}. Please check if the file exists and is not corrupted.")
    
    if not all_results:
        logger.error("No results collected, cannot proceed. Please check checkpoint files and command line arguments.")
        return

    # 2. Sort and save final result file
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
    except Exception as e:
        logger.error(f"Error saving final file: {e}", exc_info=True)
        logger.error("Error occurred, temporary checkpoint files will be skipped for safety.")
        cleanup = False # Force no cleanup on error

    # 3. Print statistical report
    logger.info("="*70 + f"\nFinal Statistical Summary for Mode '{mode}'\n" + "="*70)
    total_items = aggregated_stats.get('total_item_count', 0)
    success_items = aggregated_stats.get('successful_item_count', 0)
    failed_items = aggregated_stats.get('failed_item_count', 0)
    success_rate = (success_items / total_items * 100) if total_items > 0 else 0
    avg_success_time = (aggregated_stats.get('total_successful_time_s', 0) / success_items) if success_items > 0 else 0
    
    logger.info(f"  - Data Integrity: {success_items + failed_items} / {total_items} (Should be 100%)")
    logger.info(f"  - Success Rate: {success_rate:.2f}% ({success_items}/{total_items})")
    logger.info(f"    - Successful Samples: {success_items}")
    logger.info(f"    - Failed Samples: {failed_items}")
    logger.info(f"  - Total Time for Successful Samples: {aggregated_stats.get('total_successful_time_s', 0):.2f} s")
    logger.info(f"  - Average Time per Successful Sample: {avg_success_time:.2f} s")
    logger.info(f"  - Total Time for Failed Attempts: {aggregated_stats.get('total_wasted_time_s', 0):.2f} s")
    logger.info("="*70)

    # 4. Save or update summary file
    save_or_update_summary(
        output_dir=output_dir,
        model_name=model_name,
        mode=mode,
        new_stats=aggregated_stats,
        logger=logger
    )

    # 5. Clean up checkpoint files
    if cleanup:
        logger.info("Starting cleanup of temporary checkpoint files...")
        cleaned_count = 0
        for i in range(num_shards):
            checkpoint_path = get_checkpoint_path(output_dir, model_name, mode, i)
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                    logger.debug(f"Deleted: {checkpoint_path}")
                    cleaned_count += 1
                except OSError as e:
                    logger.error(f"Failed to delete checkpoint {checkpoint_path}: {e}")
        logger.info(f"Temporary checkpoint file cleanup completed, deleted {cleaned_count} files.")
    else:
        logger.info("Cleanup step skipped, checkpoint files are preserved.")


def main():
    parser = argparse.ArgumentParser(
        description="Merges inference checkpoints, generates final report and statistics. Used to recover from stuck runs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Name of the model to merge, e.g., 'Gemini2.5-pro'"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["cot", "direct"], 
        help="Evaluation mode, e.g., 'direct'"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="final_output", 
        help="Root output directory containing the 'checkpoints' subdirectory."
    )
    parser.add_argument(
        "--num_shards", 
        type=int, 
        required=True, 
        help="Total number of shards (processes) used in the original run."
    )
    parser.add_argument(
        "--no-cleanup", 
        action="store_true", 
        help="If set, original .checkpoint files will not be deleted."
    )

    args = parser.parse_args()
    logger = setup_logger()

    logger.info("========== Starting Checkpoint Merge and Statistics Script ==========")
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Num Shards: {args.num_shards}")
    logger.info(f"Cleanup Checkpoint Files: {not args.no_cleanup}")
    
    merge_and_finalize(
        model_name=args.model,
        mode=args.mode,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        cleanup=not args.no_cleanup,
        logger=logger
    )
    
    logger.info("========== Script Execution Completed ==========")


if __name__ == "__main__":
    main()