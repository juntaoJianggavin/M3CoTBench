# engine.py (Modified Version)

from typing import Dict, List, Any, Optional
import torch
import time
import logging
from accelerate import Accelerator

from models.model_registry import get_model
from config.config import Config

class InferenceEngine:
    """
    Inference Engine, focused on executing inference calculations on a single process.
    Implements independent retries, timing, and status tracking for each sample.
    """

    def __init__(
        self,
        model_name: str,
        config: Config,
        accelerator: Optional[Accelerator] = None,
    ):
        self.model_name = model_name
        self.config = config
        self.model_config = config.get_model_config(model_name)
        self.global_config = config.get_global_config()
        self.accelerator = accelerator or Accelerator()
        self.logger = logging.getLogger(__name__)

        # Initialize model
        model = get_model(
            model_name=model_name,
            model_config=self.model_config,
            device=self.accelerator.device,
        )
        model.eval()
        self.model = model

    def run(
        self,
        dataset: List[Dict[str, Any]], # Note: The dataset passed here is already sharded
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference on the given dataset (shard) on the current process.
        Process item by item to achieve precise timing and retries.
        """
        results = []
        
        # Initialize detailed statistics
        stats = {
            "total_successful_time_s": 0.0,
            "total_wasted_time_s": 0.0,
            "successful_item_count": 0,
            "failed_item_count": 0,
            "total_item_count": len(dataset)
        }
        
        # Get retry count from config
        max_retries_per_item = self.global_config.get("retries_per_item", 3)
        
        self.logger.info(f"[P{self.accelerator.process_index}] Started processing {len(dataset)} samples, max retries per sample: {max_retries_per_item}.")

        for i, item in enumerate(dataset):
            item_id = item.get('id', 'N/A')
            self.logger.debug(f"  -> Processing sample {i+1}/{len(dataset)} (ID: {item_id})")
            
            item_wasted_time_ns = 0
            
            for attempt in range(max_retries_per_item):
                try:
                    # --- Single attempt timing start ---
                    attempt_start_time = time.perf_counter_ns()
                    
                    with torch.no_grad():
                        # Call model generation, passing single sample data
                        prediction = self.model.generate(
                            image=item['image'],
                            prompt=item['prompt'],
                            # You can pass more model-specific arguments here
                        )
                    
                    attempt_end_time = time.perf_counter_ns()
                    # --- Single attempt timing end ---

                    # Check if API client returned an error string
                    if isinstance(prediction, str) and ("Error" in prediction or "Failed" in prediction):
                        raise RuntimeError(f"API returned explicit error message: {prediction}")

                    # --- Success logic ---
                    duration_ns = attempt_end_time - attempt_start_time
                    stats["total_successful_time_s"] += duration_ns / 1e9
                    stats["successful_item_count"] += 1
                    
                    # Count time spent on previous failed attempts for this sample as wasted time
                    stats["total_wasted_time_s"] += item_wasted_time_ns / 1e9

                    results.append({
                        "index": str(item_id),
                        "prediction": prediction,
                        "status": "success",
                        "duration_s": round(duration_ns / 1e9, 4),
                        "attempts": attempt + 1
                    })
                    
                    self.logger.debug(f"    [Success] ID: {item_id}, Time: {duration_ns / 1e9:.2f}s, Attempts: {attempt + 1}")
                    break # Success, break retry loop

                except Exception as e:
                    # --- Failure logic ---
                    attempt_end_time = time.perf_counter_ns()
                    attempt_duration_ns = attempt_end_time - attempt_start_time
                    item_wasted_time_ns += attempt_duration_ns
                    
                    self.logger.warning(f"    [Failed] ID: {item_id}, Attempt {attempt + 1}/{max_retries_per_item} failed. Error: {e}")

                    if attempt + 1 == max_retries_per_item:
                        # --- Final failure logic ---
                        stats["failed_item_count"] += 1
                        # Count time spent on all failed attempts for this sample as wasted time
                        stats["total_wasted_time_s"] += item_wasted_time_ns / 1e9
                        
                        results.append({
                            "index": str(item_id),
                            "prediction": f"Final failure, tried {max_retries_per_item} times. Last error: {e}",
                            "status": "failed",
                            "duration_s": 0.0, # Success duration is 0
                            "attempts": attempt + 1
                        })
                        self.logger.error(f"Sample (ID: {item_id}) finally failed.")
                    else:
                        # Prepare for next retry, adding exponential backoff wait
                        time.sleep(2 ** attempt)
            
            # Clear VRAM to prevent accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Return summary containing results and detailed statistics
        summary = {
            "results": results,
            "stats": stats
        }
        return summary

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()