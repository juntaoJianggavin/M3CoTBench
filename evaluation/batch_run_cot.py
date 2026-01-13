import os
import subprocess

OUTPUT_ROOT = 'output'
CACHE_ROOT = 'cache/recall'
PROMPT_PATH = 'prompt/prompt_recall.txt'
MODEL = 'gpt-4o'
NUM_THREADS = 20

for model_name in os.listdir(OUTPUT_ROOT):
    model_dir = os.path.join(OUTPUT_ROOT, model_name)
    if not os.path.isdir(model_dir):
        continue
    # 查找 *_cot.xlsx 文件
    for f in os.listdir(model_dir):
        if f.endswith('_cot.xlsx'):
            data_path = os.path.join(model_dir, f)
            cache_dir = os.path.join(CACHE_ROOT, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Processing {data_path} -> {cache_dir}")
            cmd = [
                'python', 'main.py',
                '--name', 'recall',
                '--num_threads', str(NUM_THREADS),
                '--prompt_path', PROMPT_PATH,
                '--data_path', data_path,
                '--cache_dir', cache_dir,
                '--model', MODEL
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)