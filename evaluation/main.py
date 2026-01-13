import json
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union

import requests
from tqdm import tqdm
import numpy as np

from dataset import get_dataset_by_path
from file_utils import save_output 

api_key = os.environ.get("AIMLAPI_KEY", "xxxxx")
base_url = os.environ.get("AIMLAPI_URL", "https://api.aimlapi.com/v1")
CHAT_URL = f"{base_url}/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


def query_gpt(inputs, args):
    """调用 AIML API 获取回答"""

    prompt_text = inputs["query_input"]

    data = {
        "model": args.model,  # 默认 gpt-4o，可换成 Llama 模型
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0
    }

    resp = requests.post(CHAT_URL, headers=headers, json=data)
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]

    class FakeGPTOutput:
        """构造一个模拟结构，保持原代码兼容"""
        class Choice:
            def __init__(self, text):
                self.message = type("msg", (), {"content": text})
        def __init__(self, text):
            self.choices = [FakeGPTOutput.Choice(text)]

    return FakeGPTOutput(content), inputs["index"]

# -----------------------------
# 主逻辑
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="recall", help="Path to the dataset class.")
parser.add_argument("--num_threads", type=int, default=16, help="Number of threads.")
parser.add_argument("--prompt_path", default="prompt/prompt_recall.txt", help="Path to the prompt file.")
parser.add_argument("--meta_data_path", default="/z_data/byl/MME-CoT-benchmarks/output_with_key_annotation_steps_final.xlsx", help="Path to the meta data file.")
parser.add_argument("--data_path", default="results/Llama-3.2-11B/Llama-3.2-11B_cot.xlsx", help="Path to the query input file.")
parser.add_argument("--model", type=str, default='gpt-4o')
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--cache_dir", type=str, default='cache/recall/Llama-3.2-11B')
args = parser.parse_args()


def task(inputs: Dict[str, Union[str, Dict[str, Union[str, int]]]]) -> Dict[str, Union[Dict[str, int], List[str]]]:
    global dataset_name

    if inputs.get('answer_ai'):
        result = inputs
        print(f"Answer already given for {inputs['index']}")
        result["valid_outputs"] = inputs["answer_ai"]
    else:
        try:
            print(f"Querying GPT for {inputs['index']}")
            gpt_output, index = query_gpt(inputs, args)

            result = {
                "valid_outputs": gpt_output.choices[0].message.content,
                "index": index,
            }

            if 'valid_outputs' in inputs:
                del inputs['valid_outputs']

            result.update(inputs)
            del result['query_input']
        except Exception as e:
            result = {"error_message": str(e)}
            print(result)
            return {}

    os.makedirs(args.cache_dir, exist_ok=True)
    json.dump(result, open(f'./{args.cache_dir}/{result["index"]}.json', 'w'), indent=4)
    return result


if __name__ == "__main__":

    dataset_args = {
        'prompt_path': getattr(args, "prompt_path", None),
        'data_path': getattr(args, "data_path", None),
        'cache_dir': getattr(args, "cache_dir", None),
        'image_folder': getattr(args, "image_folder", None),
        'meta_data_path': getattr(args, "meta_data_path", None),
    }

    os.makedirs(args.cache_dir, exist_ok=True)

    dataset = get_dataset_by_path(args.name, dataset_args)
    dataset_name = args.name

    results = []
    start_time = time.time()

    if args.num_threads == 0:
        print("Using single-threaded execution")
        progress = tqdm(total=len(dataset), unit="task")
        for d in dataset:
            results.append(task(d))
            progress.update(1)
        progress.close()
    else:
        print(f"Using {args.num_threads} threads")
        progress = tqdm(total=len(dataset), unit="task")
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [executor.submit(task, d) for d in dataset]
            for f in futures:
                results.append(f.result())
                progress.update(1)
        progress.close()

    duration = time.time() - start_time
    print(f"Total time: {duration:.2f}s")
