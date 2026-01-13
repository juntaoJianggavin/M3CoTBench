import json
import os
import argparse
import logging

# ==============================================================================
#  Please modify this to the actual key name used for storing time in your file!
#  For example, if you find that time is stored in the "time" key, write: TIME_KEY_IN_YOUR_JSON = "time"
# ==============================================================================
TIME_KEY_IN_YOUR_JSON = "duration_s"  # <--- Modify here! Replace with the correct key name from your file

def setup_logger():
    """Configure a simple logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Recalculate and update the summary file based on the final results JSON file."
    )
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to the JSON file containing the final (fixed) results. Example: final_output/Model/Model_cot.json"
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        required=True,
        help="Path to the summary file that needs to be updated. Example: final_output/Model/Model_summary.json"
    )
    args = parser.parse_args()
    logger = setup_logger()

    # 1. Read the final results file
    logger.info(f"Loading final results from '{args.results_file}'...")
    try:
        with open(args.results_file, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Cannot read or parse results file: {e}")
        return

    # 2. Recalculate statistics
    logger.info(f"Recalculating statistics using time key '{TIME_KEY_IN_YOUR_JSON}'...")
    stats = {
        "total_item_count": len(final_results),
        "successful_item_count": 0,
        "failed_item_count": 0,
        "total_successful_time_s": 0.0,
        "total_wasted_time_s": 0.0,
    }

    key_found_once = False
    for item in final_results:
        # Check if the time key exists
        time_value = item.get(TIME_KEY_IN_YOUR_JSON)
        if time_value is not None:
            key_found_once = True
            time_float = float(time_value)
        else:
            time_float = 0.0

        if item.get("status") == "success":
            stats["successful_item_count"] += 1
            stats["total_successful_time_s"] += time_float
        else:
            stats["failed_item_count"] += 1
            stats["total_wasted_time_s"] += time_float

    if not key_found_once:
        logger.warning(f"Warning: Time key '{TIME_KEY_IN_YOUR_JSON}' was not found in any result entry. All times will be 0.")

    logger.info("="*20 + " Recalculated Statistical Summary " + "="*20)
    logger.info(f"  - Total items: {stats['total_item_count']}")
    logger.info(f"  - Success count: {stats['successful_item_count']}")
    logger.info(f"  - Failure count: {stats['failed_item_count']}")
    logger.info(f"  - Accumulated success time: {stats['total_successful_time_s']:.2f} s")
    logger.info("="*60)

    # 3. Read and update the summary file
    logger.info(f"Preparing to update summary file: {args.summary_file}")
    if not os.path.exists(args.summary_file):
        logger.error("Summary file does not exist, creating a new one.")
        all_summaries = {}
    else:
        try:
            with open(args.summary_file, 'r', encoding='utf-8') as f:
                all_summaries = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading summary file: {e}. Creating a new one.")
            all_summaries = {}

    # Infer mode
    mode = 'cot' if '_cot.json' in args.results_file else 'direct'
    
    # Keep data for other modes in the summary file, only update the current mode
    if mode in all_summaries:
        logger.info(f"Found mode '{mode}' in summary file, overwriting with recalculated data.")
        # Preserve original wasted_time as it records the sum of historical failed attempts
        stats['total_wasted_time_s'] += all_summaries[mode].get('total_wasted_time_s', 0)
    else:
         logger.info(f"Mode '{mode}' not found in summary file, adding new statistical entry.")

    all_summaries[mode] = stats

    # 4. Write back to file
    try:
        with open(args.summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=4, ensure_ascii=False)
        logger.info("Summary file successfully updated!")
    except Exception as e:
        logger.error(f"Error occurred while writing updated summary file: {e}")

if __name__ == "__main__":
    main()