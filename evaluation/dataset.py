import json
import os
from tqdm import tqdm
import string
from direct_eval import extract_answer_from_item
from file_utils import read_results
import pandas as pd

def parse_json_field(field):
    """Safely convert a JSON string to a dict; if it's already a dict, return as is."""
    if isinstance(field, str):
        try:
            return json.loads(field)
        except Exception as e:
            print(f"[Parsing Failed] {e}, content: {field[:50]}...")
            return {}
    return field

def make_gt_dict(gt_dataset):
    gt_dict = dict()
    for i, data in enumerate(gt_dataset):
        gt_dict[data['index']] = data
    return gt_dict

# make prompt for recall, precision, relevance, reflection_quality
def make_prompt(name, c, gt_inst, prompt):
    gt_set = []
    if name == 'recall':
        cnt = 0
        # Adapt to your new data structure (flat solutionN->str)
        for f in sorted(gt_inst['key_annotation_steps']):
            step = gt_inst['key_annotation_steps'][f]
            if step is None or str(step).strip() == '':
                continue
            gt_set.append(dict(
                step_index=cnt,
                content=str(step).strip()
            ))
            cnt += 1
    else:
        gt_set = []
        for f in gt_inst['key_annotation_steps']:
            step = gt_inst['key_annotation_steps'][f]
            if step is None or str(step).strip() == '':
                continue
            gt_set.append(str(step).strip())
        gt_set = [l.strip() for l in gt_set if l.strip() != '']
    # make question
    question = gt_inst['question']
    options = {
            cand: gt_inst[cand]
            for cand in string.ascii_uppercase
            if cand in gt_inst and not gt_inst[cand] != None and gt_inst[cand] != ''
        }
    options_prompt = 'Options:\n'
    for key, item in options.items():
        options_prompt += f'{key}. {item}\n'
    if len(options) != 0:
        question += options_prompt
    # add information
    prompt = prompt.format(
        question=c['question'],
        answer=gt_inst['answer'],
        solution=c['prediction'],
        gt_annotation=json.dumps(gt_set)
    )
    return prompt


def get_dataset_by_path(name, dataset_args):
    # load dataset
    gt_dataset = pd.read_excel(dataset_args["meta_data_path"])
    gt_dataset = gt_dataset.to_dict(orient='records')  # Convert to list[dict, dict, ...]

    # Convert key_annotation_steps field to dict
    for item in gt_dataset:
        if 'key_annotation_steps' in item:
            item['key_annotation_steps'] = parse_json_field(item['key_annotation_steps'])

    gt_dataset_dict = make_gt_dict(gt_dataset)

    # load all the result and its index
    results = read_results(dataset_args["data_path"]) # read either from xlsx or json
    print(f"Load {len(results)} results from {dataset_args['data_path']}")
    
    # filter what have already collected in cache
    cached_index = []
    for file in os.listdir(dataset_args['cache_dir']):
        cached_index.append(int(os.path.splitext(file)[0]))
    filtered_results = []
    for c in results:
        if int(c['index']) not in cached_index:
            filtered_results.append(c)
    results = filtered_results
    
    # read the prompt
    with open(dataset_args["prompt_path"], 'r') as f:
        prompt = f.read().strip()
    
    if name in [
        'recall',
        'precision'
    ]:
        return_list = []
        for c in tqdm(results):
            gt_inst = gt_dataset_dict[c['index']]
            # Ensure key_annotation_steps is a dict again (in case it wasn't handled elsewhere)
            gt_inst['key_annotation_steps'] = parse_json_field(gt_inst['key_annotation_steps'])
            c['key_annotation_steps'] = gt_inst['key_annotation_steps']

            # this is for judge task
            c["answer_ai"] = None

            c['query_input'] = [
                {"type": "text", "text": make_prompt(name, c, gt_inst, prompt)}
            ]

            c['index'] = c['index']
            return_list.append(c)
    else:
        raise NotImplementedError
    
    return return_list