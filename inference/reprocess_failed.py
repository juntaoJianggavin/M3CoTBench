import os
import sys
import argparse
import json
import logging
import time

# Ensure project root is in Python path to import other modules
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from config.config import Config
from data.dataset import MedicalVQADataset
from inference.engine import InferenceEngine
from utils.logger import setup_logger
from accelerate import Accelerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reprocess samples that failed or had empty predictions during the main evaluation flow.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # ... (Content of this function unchanged, omitted for brevity)
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the JSON result file containing failed samples (e.g., final_output/Qwen-VL-Chat/Qwen-VL-Chat_cot.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name used for re-inference (must match the original model)."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./dataset/M3CoTBench.xlsx",
        help="Path to the original dataset file."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./dataset/images",
        help="Path to the original image directory."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save the new processing results.\nDefaults to creating 'reprocessed_results.json' in the same directory as the input file."
    )
    parser.add_argument(
        "--update-in-place",
        action="store_true",
        help="If set, this option will directly update newly successful results back into the original --input-file."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional configuration file path.")
    parser.add_argument("--pre-prompt", type=str, default="", help="Optional pre-prompt.")
    parser.add_argument("--post-prompt", type=str, default="", help="Optional post-prompt.")
    
    return parser.parse_args()


# --- MODIFICATION START: Function signature and internal logic rewritten ---
def update_summary_file(args, new_stats, final_results, logger):
    """
    Read, completely recalculate based on final results, and save the statistical summary file.

    Args:
        args (argparse.Namespace): Object containing all command line arguments.
        new_stats (dict): Statistics obtained from this reprocessing run (mainly for calculating wasted time).
        final_results (list): Complete list of results containing final statuses for all samples.
        logger (logging.Logger): Logger instance.
    """
    # --- 1. Identify and read summary file ---
    model_output_dir = os.path.dirname(args.input_file)
    safe_model_name = args.model.replace("/", "_").replace("\\", "_").replace(":", "_")
    summary_filename = f"{safe_model_name}_summary.json"
    summary_filepath = os.path.join(model_output_dir, summary_filename)

    logger.info(f"Preparing to update statistical summary file: {summary_filepath}")

    if not os.path.exists(summary_filepath):
        logger.warning(f"Summary file {summary_filepath} does not exist. Cannot update.")
        return
        
    try:
        with open(summary_filepath, 'r', encoding='utf-8') as f:
            all_summaries = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error reading or parsing summary file {summary_filepath}: {e}. Cannot update.")
        return

    # --- 2. Recalculate statistics based on final results ---
    logger.info("Recalculating statistics based on the final complete result list...")
    
    mode = 'cot' if '_cot.json' in args.input_file else 'direct'
    if mode not in all_summaries:
        logger.error(f"Statistics for mode '{mode}' not found in summary file {summary_filepath}. Cannot update.")
        return
        
    old_stats = all_summaries[mode]
    total_item_count = old_stats.get('total_item_count')
    if total_item_count is None:
        logger.error("Missing 'total_item_count' in summary file, cannot accurately calculate failure count.")
        total_item_count = len(final_results) # Fallback solution

    recalculated_successful_count = 0
    recalculated_successful_time_s = 0.0

    for item in final_results:
        # Define a valid success: status is 'success' and 'prediction' field has actual content
        if item.get('status') == 'success' and item.get('prediction'):
            recalculated_successful_count += 1
            recalculated_successful_time_s += item.get('duration_s', 0)

    recalculated_failed_count = total_item_count - recalculated_successful_count
    
    # For "wasted time", the logic remains: it is the initial run's wasted time + time wasted in this rerun where it still failed.
    # This is a meaningful metric reflecting total resources consumed by the model on all unsuccessful attempts.
    updated_wasted_time = old_stats.get('total_wasted_time_s', 0) + new_stats.get('total_wasted_time_s', 0)

    logger.info("Recalculation complete:")
    logger.info(f"  - Mode: {mode}")
    logger.info(f"  - Total items: {total_item_count}")
    logger.info(f"  - Recalculated success count: {recalculated_successful_count}")
    logger.info(f"  - Recalculated failure count: {recalculated_failed_count}")
    logger.info(f"  - Recalculated total success time: {recalculated_successful_time_s:.2f} s")
    logger.info(f"  - Updated total wasted time: {updated_wasted_time:.2f} s")

    # --- 3. Update and write back to summary file ---
    all_summaries[mode]['successful_item_count'] = recalculated_successful_count
    all_summaries[mode]['failed_item_count'] = recalculated_failed_count
    all_summaries[mode]['total_successful_time_s'] = recalculated_successful_time_s
    all_summaries[mode]['total_wasted_time_s'] = updated_wasted_time

    try:
        temp_file = summary_filepath + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=4, ensure_ascii=False)
        os.replace(temp_file, summary_filepath)
        logger.info(f"Statistical summary file successfully updated: {summary_filepath}")
    except Exception as e:
        logger.error(f"Error occurred while writing updated summary file: {e}")
# --- MODIFICATION END ---


def main():
    args = parse_args()
    logger = setup_logger()
    
    # --------------------------------------------------------------------------
    # 1. Load and filter samples needing reprocessing
    # --------------------------------------------------------------------------
    logger.info(f"Loading results from '{args.input_file}'...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Cannot read or parse input file: {e}")
        return

    items_to_reprocess = [
        item for item in all_results 
        if item.get("status") == "failed" or 
           (item.get("status") == "success" and not item.get("prediction"))
    ]
    
    if not items_to_reprocess:
        logger.info("No samples with status 'failed' or empty prediction found in file. No processing needed.")
        return

    logger.info(f"Found {len(items_to_reprocess)} samples with status 'failed' or empty prediction needing reprocessing.")
    reprocess_indices = {item['index'] for item in items_to_reprocess}

    # --------------------------------------------------------------------------
    # 2. Reconstruct dataset for failed samples from original data source
    # --------------------------------------------------------------------------
    # ... (Content of this section unchanged, omitted for brevity)
    logger.info("Loading dataset from original data source to match samples needing reprocessing...")
    
    mode = 'cot' if '_cot.json' in args.input_file else 'direct'
    logger.info(f"Inferred mode from filename: '{mode}'")

    full_dataset = MedicalVQADataset(
        data_path=args.data_path,
        image_dir=args.image_dir,
        mode=mode,
        pre_prompt=args.pre_prompt,
        post_prompt=args.post_prompt
    )
    
    dataset_to_reprocess = []
    for i in range(len(full_dataset)):
        sample = full_dataset[i]
        if str(sample['id']) in reprocess_indices:
            dataset_to_reprocess.append(sample)

    if len(dataset_to_reprocess) != len(items_to_reprocess):
        logger.warning("Warning: Number of pending samples found in original dataset does not match those in the JSON file.")
    
    logger.info(f"Successfully constructed dataset with {len(dataset_to_reprocess)} pending samples.")

    # --------------------------------------------------------------------------
    # 3. Initialize and run inference engine
    # --------------------------------------------------------------------------
    # ... (Content of this section unchanged, omitted for brevity)
    config = Config(args.config) if args.config else Config()
    accelerator = Accelerator()

    logger.info(f"Initializing inference engine for model '{args.model}'...")
    engine = InferenceEngine(
        model_name=args.model,
        config=config,
        accelerator=accelerator
    )
    
    logger.info("Starting re-inference...")
    start_time = time.time()
    
    summary = engine.run(dataset=dataset_to_reprocess)
    
    end_time = time.time()
    logger.info(f"Re-inference completed, duration: {end_time - start_time:.2f} seconds.")

    newly_processed_results = summary.get("results", [])
    new_stats = summary.get("stats", {})

    logger.info("="*30 + " Reprocessing Result Summary " + "="*30)
    logger.info(f"  - Attempted samples: {new_stats.get('total_item_count', 0)}")
    logger.info(f"  - Current success count: {new_stats.get('successful_item_count', 0)}")
    logger.info(f"  - Current failure count: {new_stats.get('failed_item_count', 0)}")
    logger.info(f"  - Accumulated success time: {new_stats.get('total_successful_time_s', 0):.2f} s")
    logger.info("="*80)

    # --------------------------------------------------------------------------
    # 4. Save and merge results
    # --------------------------------------------------------------------------
    if not newly_processed_results:
        logger.warning("No new results generated, exiting program.")
        return

    # --- MODIFICATION START: Logic for constructing final result list moved up ---
    # Regardless of in-place update, we need a final, merged result list for statistics.
    logger.info("Merging original results with newly processed results...")
    new_results_map = {item['index']: item for item in newly_processed_results}
    
    final_results = []
    updated_count = 0
    for item in all_results:
        new_item = new_results_map.get(item['index'])
        # Only use the new result if it is valid (success and has prediction)
        if new_item and new_item.get('status') == 'success' and new_item.get('prediction'):
            final_results.append(new_item)
            updated_count += 1
        else:
            final_results.append(item)
    
    final_results.sort(key=lambda x: int(x.get('index', 0)))
    logger.info(f"Result merge completed, updated {updated_count} entries.")
    # --- MODIFICATION END ---

    output_file = args.output_file
    if not output_file:
        base_dir = os.path.dirname(args.input_file)
        output_file = os.path.join(base_dir, "reprocessed_results.json")

    logger.info(f"Saving {len(newly_processed_results)} newly processed results (current run only) to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(newly_processed_results, f, indent=2, ensure_ascii=False)
    logger.info("Save completed.")

    if args.update_in_place:
        logger.info(f"Detected --update-in-place argument, updating original file '{args.input_file}' with merged final results...")
        try:
            temp_file = args.input_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, args.input_file)
            logger.info(f"Original file successfully updated.")
        except Exception as e:
            logger.error(f"Error occurred while updating original file: {e}")

    # --------------------------------------------------------------------------
    # 5. Update statistical summary file
    # --------------------------------------------------------------------------
    # --- MODIFICATION START: Passing final_results to function ---
    update_summary_file(args, new_stats, final_results, logger)
    # --- MODIFICATION END ---


if __name__ == "__main__":
    main()