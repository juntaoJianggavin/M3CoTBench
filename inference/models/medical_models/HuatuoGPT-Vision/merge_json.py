import json
import os
import glob

def filter_condition(item):
    """
    定义筛选条件。
    如果条目符合保留条件，则返回 True；否则返回 False。
    
    *** 请在此处自定义您的筛选逻辑 ***

    Args:
        item (dict): 一个数据条目，例如 {"index": "1", "prediction": "..."}

    Returns:
        bool: True 表示保留该条目，False 表示丢弃。
    """
    # --- 示例筛选逻辑 ---
    # 您可以根据需要修改或替换这里的逻辑。

    # 示例 1: 保留 prediction 中包含特定关键字的条目
    # keyword = "Visual acuity test"
    # if 'prediction' in item and isinstance(item['prediction'], str):
    #     return keyword.lower() in item['prediction'].lower()
    # return False

    # 示例 2: 保留 index 小于 50 的条目
    try:
        # 尝试将 index 转换为整数进行比较
        if 'index' in item and int(item['index']) < 50:
            return True
    except (ValueError, TypeError):
        # 如果 index 不是有效的数字，则不保留
        return False
    return False
    
    # 示例 3: 如果您不想进行任何筛选，直接返回 True
    # return True


def merge_filter_sort_json(input_path, output_file, perform_filter=True):
    """
    合并、筛选并排序指定路径下的所有JSON文件。

    Args:
        input_path (str): 包含JSON文件的目录路径。
        output_file (str): 合并后输出的JSON文件名。
        perform_filter (bool): 是否执行筛选操作。
    """
    merged_data = []
    seen_indices = set()

    json_files = sorted(glob.glob(os.path.join(input_path, '*.json')))

    if not json_files:
        print(f"在目录 '{input_path}' 中没有找到任何 JSON 文件。")
        return

    print(f"找到以下文件进行处理: {', '.join(os.path.basename(f) for f in json_files)}")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if not isinstance(data, list):
                    print(f"警告: 文件 '{os.path.basename(file_path)}' 的内容不是一个列表，已跳过。")
                    continue

                for item in data:
                    if not (isinstance(item, dict) and 'index' in item):
                        print(f"警告: 文件 '{os.path.basename(file_path)}' 中发现格式不正确的条目，已跳过: {item}")
                        continue

                    # --- 步骤 1: 筛选 ---
                    if perform_filter and not filter_condition(item):
                        continue  # 如果不满足筛选条件，则跳过此条目

                    # --- 步骤 2: 去重 ---
                    index = item['index']
                    if index not in seen_indices:
                        merged_data.append(item)
                        seen_indices.add(index)
                    else:
                        print(f"冲突发现: 索引 '{index}' 已存在。将忽略文件 '{os.path.basename(file_path)}' 中的此条目。")

        except json.JSONDecodeError:
            print(f"错误: 文件 '{os.path.basename(file_path)}' 不是有效的JSON格式，已跳过。")
        except Exception as e:
            print(f"处理文件 '{os.path.basename(file_path)}' 时发生错误: {e}")

    # --- 步骤 3: 排序 ---
    print("\n所有文件处理完毕。现在开始对合并后的数据进行排序...")
    try:
        # 使用 lambda 函数将 'index' 字符串转为整数进行排序
        merged_data.sort(key=lambda x: int(x['index']))
        print("数据已根据 'index' 字段完成升序排序。")
    except (ValueError, KeyError) as e:
        print(f"警告: 排序时发生错误，某些 'index' 值可能不是有效的数字。数据将保持合并时的顺序。错误: {e}")

    # --- 步骤 4: 保存到文件 ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"\n处理成功！结果已保存到 '{output_file}'。")
        print(f"总共筛选并合并了 {len(merged_data)} 个唯一的条目。")
    except Exception as e:
        print(f"写入输出文件 '{output_file}' 时发生错误: {e}")

# --- 使用说明 ---
if __name__ == "__main__":
    # 1. 设置包含JSON文件的目录路径
    input_directory = '/home/tione/notebook/byl/MME-CoT-benchmarks/Med-CoT-Eval/pretrain/HuatuoGPT-Vision/temp_results_final/direct' 

    # 2. 设置合并后输出的文件名
    output_filename = '/home/tione/notebook/byl/MME-CoT-benchmarks/Med-CoT-Eval/final_output/HuatuoGPT-Vision/HuatuoGPT-Vision_direct.json'

    APPLY_FILTERING = False
    merge_filter_sort_json(input_directory, output_filename, perform_filter=APPLY_FILTERING)

    # 1. 设置包含JSON文件的目录路径
    input_directory = '/home/tione/notebook/byl/MME-CoT-benchmarks/Med-CoT-Eval/pretrain/HuatuoGPT-Vision/temp_results_final/cot' 

    # 2. 设置合并后输出的文件名
    output_filename = '/home/tione/notebook/byl/MME-CoT-benchmarks/Med-CoT-Eval/final_output/HuatuoGPT-Vision/HuatuoGPT-Vision_cot.json'
    APPLY_FILTERING = False
    merge_filter_sort_json(input_directory, output_filename, perform_filter=APPLY_FILTERING)