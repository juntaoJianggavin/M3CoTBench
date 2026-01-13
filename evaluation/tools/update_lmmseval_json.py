import os
import json
import copy
import pandas as pd
import argparse

def find_all_json_files(root):
    """Recursively find all json files and return a list of relative paths"""
    json_files = []
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith('.json'):
                full_path = os.path.join(dirpath, file)
                rel_path = os.path.relpath(full_path, root)
                json_files.append(rel_path)
    return json_files

parser = argparse.ArgumentParser()
parser.add_argument("--lmms_eval_json_path", type=str, default="../inference/final_output", help="Path to the evaluation JSON file or directory.")
parser.add_argument("--save_path", type=str, default="results", help="Path to save the Excel file or directory.")
parser.add_argument("--dataset_path", type=str, default="./output_with_key_annotation_steps_final.xlsx", help="Path to the local dataset file (CSV or XLSX).")
parser.add_argument("--index_column", type=str, default="index", help="Column name to use as index in the dataset file.")
args = parser.parse_args()

# Load local dataset (supports csv and xlsx)
if args.dataset_path.endswith('.csv'):
    dataset_df = pd.read_csv(args.dataset_path)
elif args.dataset_path.endswith('.xlsx'):
    dataset_df = pd.read_excel(args.dataset_path)
else:
    raise ValueError("Unsupported dataset file type. Please use CSV or XLSX.")

dataset_df["index"] = [int(i + 1) for i in range(len(dataset_df))]

# Use the specified column as the index and convert to dictionary format
dataset_dict = {str(row[args.index_column]): row.to_dict() for _, row in dataset_df.iterrows()}

# Recursively find all json files
if os.path.isdir(args.lmms_eval_json_path):
    all_json_files = find_all_json_files(args.lmms_eval_json_path)
else:
    all_json_files = [os.path.basename(args.lmms_eval_json_path)]

for rel_json_path in all_json_files:
    json_path = os.path.join(args.lmms_eval_json_path, rel_json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    # Unify format: Ensure pred_data is a list and elements are dicts, otherwise try to fix
    if isinstance(pred_data, dict):
        pred_data = [pred_data]
    elif isinstance(pred_data, list):
        # Filter out non-dict elements
        pred_data = [item for item in pred_data if isinstance(item, dict)]
    else:
        print(f"[Warning] Unexpected JSON root type {type(pred_data)} in {json_path}, skipping file.")
        continue

    new_dataset = copy.deepcopy(dataset_dict)
    for i, data in enumerate(pred_data):
        try:
            index = str(data['index'])
            if index in new_dataset:
                new_dataset[index]['prediction'] = data.get('prediction', None)
            else:
                print(f"Warning: Index {index} not found in dataset (file {rel_json_path})")
        except Exception as e:
            print(f"Error processing entry {i} in {rel_json_path}: {e}")
            # You can choose to skip here, or insert debugging
            # import pdb; pdb.set_trace()
            continue

    df = pd.DataFrame.from_dict(new_dataset, orient='index')
    save_file = os.path.join(args.save_path, rel_json_path.replace('.json', '.xlsx'))
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df.to_excel(save_file, index=False)
    print(f"Saved: {save_file}")

print("Processing complete!")