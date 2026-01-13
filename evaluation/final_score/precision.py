import json 
import re
import os
import argparse
from collections import defaultdict
from json_repair import repair_json
from datasets import load_dataset
import pandas as pd
# Load dataset for category information
# === 读取本地 xlsx 数据集，建立 index -> modality 映射 ===
def load_modality_mapping(xlsx_path):
    df = pd.read_excel(xlsx_path)
    # index必须唯一
    mapping = {row['index']: row['modality'] for _, row in df.iterrows()}
    return mapping

# === 从文本中提取并修复json字符串 ===
def extract_json_string(text):
    if not text:
        return None
    text = text.replace('\\', '\\\\')
    start = text.find('[')
    if start == -1:
        return None
    stack = []
    end = -1
    for i in range(start, len(text)):
        if text[i] == '[':
            stack.append(i)
        elif text[i] == ']':
            if stack:
                stack.pop()
                if not stack:
                    end = i
                    break
    if end == -1:
        return None
    json_str = text[start:end + 1]
    return repair_json(json_str)

def parse_args():
    parser = argparse.ArgumentParser(description='calculate precision')
    parser.add_argument('--cache_dir', type=str, 
                       default='./cache/precision',
                       help='cache directory')
    parser.add_argument('--save_path', type=str,
                       default='./final_results',
                       help='output directory')
    args = parser.parse_args()
    return args

def extract_json_string(text):
    """
    Extract and process JSON string from text.
    Returns None if invalid format.
    """
    if not text:
        return None
    text = text.replace('\\', '\\\\')
    start = text.find('[')
    if start == -1:
        return None
    stack = []
    end = -1
    for i in range(start, len(text)):
        if text[i] == '[':
            stack.append(i)
        elif text[i] == ']':
            if stack:
                stack.pop()
                if not stack:
                    end = i
                    break
    if end == -1:
        return None
    json_str = text[start:end + 1]
    return repair_json(json_str)

def calculate_precision(data_list, step_type=None):
    """
    Calculate precision and ratio for given data list.
    Returns precision score and match/reasonable ratio.
    """
    try:
        match_count = 0
        reasonable_count = 0
        filtered_data = [item for item in data_list if not step_type or item['step_type'] == step_type]
        total = len(filtered_data)
        
        if total == 0:
            return 0, None
            
        for item in filtered_data:
            judgment = item['judgment']
            if judgment == 'Match':
                match_count += 1
        precision = (match_count) / total
        ratio = match_count / (match_count) if match_count > 0 else None
        
        return precision, ratio
    except Exception as e:
        return 0, None

def analyze_precision(json_file_path,modality_mapping):
    """
    Analyze precision metrics for a single json file.
    Returns metrics by type and category.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get category info from index
        index = data.get('index')
        modality = modality_mapping.get(index, None)
            
        json_str = extract_json_string(data.get('valid_outputs'))
        if not json_str:
            raise ValueError('Missing or invalid valid_outputs field')
            
        steps_data = json.loads(json_str, strict=False)
        if not steps_data:
            raise ValueError('Empty steps_data')
        
        # Calculate overall metrics
        target_types = ['modality or examination types','image feature description','key conclusions','additional analysis']
        filtered_data = [item for item in steps_data if item['step_type'] in target_types]
        overall_precision, overall_ratio = calculate_precision(filtered_data)
        
        # Calculate metrics by type
        type_metrics = {}
        for step_type in target_types:
            prec, ratio = calculate_precision(steps_data, step_type)
            type_metrics[step_type] = {
                'score': prec,
                'ratio': ratio
            }
        
        return {
            'success': True,
            'precision': overall_precision,
            'ratio': overall_ratio,
            'modality': modality,
            'type_metrics': type_metrics 
        }
    except json.JSONDecodeError as e:
        return {'success': False, 'error': f'JSON decode error: {str(e)}'}
    except ValueError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        return {'success': False, 'error': f'Unknown error: {str(e)}'}

def process_model_files(model_path, modality_mapping):
    """Process all files in model directory and aggregate results"""
    results = {
        'precisions': [],
        'ratios': [],
        'error_files': [],
        'type_metrics': defaultdict(lambda: {'scores': [], 'ratios': []}),
        'modality_metrics': defaultdict(list),
    }
    
    for json_file in os.listdir(model_path):
        if not json_file.endswith('.json'):
            continue
            
        file_path = os.path.join(model_path, json_file)
        result = analyze_precision(file_path,modality_mapping)
        
        if result.get('success'):
            # Collect overall metrics
            results['precisions'].append(result['precision'])
            if result['ratio'] is not None:
                results['ratios'].append(result['ratio'])
            
            # Collect type metrics
            for step_type, metrics in result['type_metrics'].items():
                results['type_metrics'][step_type]['scores'].append(metrics['score'])
                if metrics['ratio'] is not None:
                    results['type_metrics'][step_type]['ratios'].append(metrics['ratio'])
        else:
            results['error_files'].append({
                'file': json_file,
                'error': result.get('error', 'Unknown error')
            })
    
    return results

def process_all_models(cache_dir, save_path, modality_mapping):
    """Process all models and save aggregated results"""
    model_stats = {}
    results_data = {}
    all_error_files = {}
    
    # Create output directory
    save_dir = os.path.join(save_path, 'precision')
    os.makedirs(save_dir, exist_ok=True)
    
    for model in os.listdir(cache_dir):
        model_path = os.path.join(cache_dir, model)
        if not os.path.isdir(model_path):
            continue
            
        results = process_model_files(model_path, modality_mapping)
        model_stats[model] = results
        
        if results['error_files']:
            all_error_files[model] = results['error_files']
        
        # Calculate averages
        model_results = {
            "overall_metrics": {
                "average_precision": round(sum(results['precisions'])/len(results['precisions']), 4) if results['precisions'] else 0,
                "average_ratio": round(sum(results['ratios'])/len(results['ratios']), 4) if results['ratios'] else None
            },
            "modality": {
                mod: round(sum(vals) / len(vals), 4) if vals else None
                for mod, vals in results['modality_metrics'].items()
            },
            "type_metrics": {},
            "category": {},
            "subcategory": {}
        }
        
        # Add type metrics
        for step_type, metrics in results['type_metrics'].items():
            scores = metrics['scores']
            ratios = metrics['ratios']
            if scores:
                model_results["type_metrics"][step_type] = {
                    "average_precision": round(sum(scores)/len(scores), 4),
                    "average_ratio": round(sum(ratios)/len(ratios), 4) if ratios else None
                }
        
        results_data[model] = model_results

    # Save main results
    output_file = os.path.join(save_dir, 'precision_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    # Save error file if exists
    if all_error_files:
        error_file = os.path.join(save_dir, 'precision_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(all_error_files, f, indent=4, ensure_ascii=False)

    return model_stats

if __name__ == '__main__':
    modality_mapping = load_modality_mapping('/z_data/byl/MME-CoT-benchmarks/output_with_key_annotation_steps_final.xlsx')
    args = parse_args()
    process_all_models(args.cache_dir, args.save_path, modality_mapping)
    print('Done')