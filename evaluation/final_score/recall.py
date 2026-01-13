import json
import os
import argparse
import pandas as pd
from json_repair import repair_json
from collections import defaultdict

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

# === 单个样本recall分析 ===
def analyze_recall(json_file_path, modality_mapping):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        index = data.get('index')
        modality = modality_mapping.get(index, None)
        gt_data = data.get('key_annotation_steps', {})

        # 统计所有非空solution的数量
        gt_steps = [v for v in gt_data.values() if v is not None and str(v).strip() != '']
        total_count = len(gt_steps)
        if total_count == 0:
            raise ValueError('No valid ground truth steps found')

        # 解析模型输出
        json_str = extract_json_string(data.get('valid_outputs'))
        if not json_str:
            raise ValueError('valid_outputs not found or invalid')
        steps_data = json.loads(json_str, strict=False)
        if len(steps_data) < total_count:
            # 不足也继续
            steps_data = steps_data + [{} for _ in range(total_count - len(steps_data))]
        steps_data = steps_data[:total_count]

        # 统计recall和匹配数
        matched = sum(1 for step in steps_data if step.get('judgment') == 'Matched')
        recall = matched / total_count if total_count else None
    
        data['recall'] = recall
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return {
            'success': True,
            'recall': recall,
            'modality': modality,
            'type_metrics': defaultdict(lambda: {'scores': [], 'ratios': []}),
            'matched_count': matched  # 新增返回值
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# === 批量处理一个模型的所有文件 ===
def process_model_files(model_path, modality_mapping):
    results = {
        'recall': [],
        'modality_metrics': defaultdict(list),
        'error_files': [],
        "type_metrics": {},
        'total_matched_steps': 0  # 新增累计字段
    }
    for step_type, metrics in results['type_metrics'].items():
            scores = metrics['scores']
            ratios = metrics['ratios']
            if scores:
                results["type_metrics"][step_type] = {
                    "average_precision": round(sum(scores)/len(scores), 4),
                    "average_ratio": round(sum(ratios)/len(ratios), 4) if ratios else None
                }
    for json_file in os.listdir(model_path):
        if not json_file.endswith('.json'):
            continue
        json_file_path = os.path.join(model_path, json_file)
        result = analyze_recall(json_file_path, modality_mapping)
        if result['success']:
            if result['recall'] is not None:
                results['recall'].append(result['recall'])
                results['total_matched_steps'] += result.get('matched_count', 0)
                if result['modality']:
                    results['modality_metrics'][result['modality']].append(result['recall'])
        else:
            results['error_files'].append({'file': json_file, 'error': result.get('error', '未知错误')})
    return results

# === 汇总所有模型 ===
def process_all_models(cache_dir, save_path, modality_mapping):
    model_stats = {}
    results_data = {}
    all_error_files = {}
    save_dir = os.path.join(save_path, 'recall')
    os.makedirs(save_dir, exist_ok=True)

    for model in os.listdir(cache_dir):
        model_path = os.path.join(cache_dir, model)
        if not os.path.isdir(model_path):
            continue
        results = process_model_files(model_path, modality_mapping)
        model_stats[model] = results
        if results['error_files']:
            all_error_files[model] = results['error_files']

        # 汇总平均及总匹配步骤数
        model_results = {
            "overall_metrics": {
                "average_recall": round(sum(results['recall']) /len(results['recall']), 4) if results['recall'] else None,
                "total_matched_steps": results.get('total_matched_steps', 0)
            },
            "modality": {
                mod: round(sum(vals) / len(vals), 4) if vals else None
                for mod, vals in results['modality_metrics'].items()
            },
            "type_metrics": {},
            "category": {},
            "subcategory": {}
        }
        results_data[model] = model_results

    output_file = os.path.join(save_dir, 'recall_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    if all_error_files:
        error_file = os.path.join(save_dir, 'recall_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(all_error_files, f, indent=4, ensure_ascii=False)

    return model_stats

# === 命令行主入口 ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate recall (single type, with modality)')
    parser.add_argument('--cache_dir', type=str, default='./cache/recall', help='cache directory')
    parser.add_argument('--save_path', type=str, default='./final_results', help='output directory')
    parser.add_argument('--xlsx_path', type=str, default='/z_data/byl/MME-CoT-benchmarks/output_with_key_annotation_steps_final.xlsx', help='path to your xlsx')
    args = parser.parse_args()

    modality_mapping = load_modality_mapping(args.xlsx_path)
    process_all_models(args.cache_dir, args.save_path, modality_mapping)
    print('Done')
