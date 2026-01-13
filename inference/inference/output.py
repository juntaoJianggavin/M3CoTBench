import os
import json
import datetime
from typing import Dict, List, Any, Optional
import base64
import io
from PIL import Image
import pandas as pd
from loguru import logger as eval_logger

def save_predictions(
    results: List[Dict[str, Any]],
    model_name: str,
    mode: str,
    output_dir: str,
    shard_index: Optional[int] = None
) -> str:
    """
    Save prediction results to a file.
    
    Args:
        results: List of prediction results.
        model_name: Name of the model.
        mode: Inference mode.
        output_dir: Output directory.
        shard_index: Index of the shard (optional).
        
    Returns:
        Path to the output file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Format timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add shard_index to filename if present
    if shard_index is not None:
        filename = f"{model_name}_{mode}_shard{shard_index}_{timestamp}.json"
    else:
        filename = f"{model_name}_{mode}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    

    # Prepare structure to save
    submission_data = []
    for result in results:
        submission_data.append({
            "index": result.get("id", "unknown"),
            "prediction": result.get("prediction", "")
        })
        
    # Write results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
        
    print(f"Predictions saved to: {output_path}")
    return output_path

def format_results_for_evaluation(
    results_path: str,
    reference_data_path: str,
    output_format: str = "json",
    use_gpt_eval: bool = False,
    api_client = None
) -> Dict[str, Any]:
    """
    Format results for evaluation.
    
    Args:
        results_path: Path to the results file.
        reference_data_path: Path to the reference data file.
        output_format: Output format.
        use_gpt_eval: Whether to use GPT for evaluation.
        api_client: API client for GPT evaluation.
        
    Returns:
        Formatted results.
    """
    # Load results
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Load reference data
    with open(reference_data_path, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
        if isinstance(reference_data, dict) and "data" in reference_data:
            reference_data = reference_data["data"]
    
    # Build mapping from index to reference data
    reference_map = {}
    for item in reference_data:
        if "index" in item:
            reference_map[item["index"]] = item
    
    # Add evaluation results
    eval_results = []
    
    for result in results:
        item_index = result.get("index", "unknown")
        prediction = result.get("prediction", "")
        
        if item_index in reference_map:
            reference_item = reference_map[item_index]
            question = reference_item.get("question", "")
            answer = reference_item.get("answer", "")
            
            eval_item = {
                "index": item_index,
                "question": question,
                "answer": answer,
                "prediction": prediction
            }
            
            # If using GPT for evaluation
            if use_gpt_eval and api_client:
                from .gpt_evaluator import evaluate_prediction
                question_data = {
                    "index": item_index, 
                    "question": question, 
                    "answer": answer, 
                    "response": prediction
                }
                eval_score = evaluate_prediction(question_data, api_client)
                eval_item["gpt_score"] = eval_score
            
            eval_results.append(eval_item)
    
    # Output according to specified format
    if output_format == "json":
        formatted_results = {
            "eval_results": eval_results,
            "summary": {
                "total_samples": len(eval_results),
                "correct_samples": sum(1 for r in eval_results if r.get("gpt_score", 0) == 1) if use_gpt_eval else None
            }
        }
        return formatted_results
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def save_summary(
    summaries: List[Dict[str, Any]],
    output_dir: str
) -> str:
    """
    Save inference summary.
    
    Args:
        summaries: List of summaries.
        output_dir: Output directory.
        
    Returns:
        Path to the summary file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Format timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build output filename
    filename = f"inference_summary_{timestamp}.json"
    logdir= os.path.join(output_dir, "logs")
    os.makedirs(logdir, exist_ok=True)
    output_path = os.path.join(logdir, filename)
    
    # Write summary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "summaries": summaries
        }, f, ensure_ascii=False, indent=2)
        
    print(f"Summary saved to: {output_path}")
    return output_path

def generate_submission_file(filename, args):
    """
    Generate submission file path.
    
    Args:
        filename: Filename.
        args: Dictionary of arguments.
        
    Returns:
        Full file path.
    """
    # Ensure output directory exists
    output_dir = args.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Return full path
    return os.path.join(output_dir, filename)

def encode_image_to_base64(image):
    """
    Encode image to base64.
    
    Args:
        image: PIL image.
        
    Returns:
        Base64 encoded string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")