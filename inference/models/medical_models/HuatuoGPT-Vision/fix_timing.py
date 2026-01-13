#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from datetime import datetime

# --- 配置日志记录 ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_existing_timing(timing_file):
    """
    加载现有的计时文件 (从 eval_new.py 复制而来)
    """
    if os.path.exists(timing_file):
        try:
            with open(timing_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"读取现有计时文件失败: {e}，将创建新文件")
    
    return {
        "model_name": "",
        "modes": {},
        "summary": {},
        "last_updated": ""
    }

def save_timing_results(model_name, mode_times, output_dir):
    """
    保存计时结果到JSON文件 (从 eval_new.py 复制并已修复)
    """
    timing_file = os.path.join(output_dir,"timing_results", "HuatuoGPT-Vision.json")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    timing_data = load_existing_timing(timing_file)
    
    if not timing_data["model_name"]:
        timing_data["model_name"] = model_name
    elif timing_data["model_name"] != model_name:
        logger.warning(f"模型名称不匹配！现有: {timing_data['model_name']}, 当前: {model_name}")
    
    for mode, time_seconds in mode_times.items():
        if time_seconds > 0:
            timing_data["modes"][mode] = {
                "dataset_total_time_seconds": round(time_seconds, 2),
                "dataset_total_time_minutes": round(time_seconds / 60, 2),
                "dataset_total_time_hours": round(time_seconds / 3600, 2),
                "updated_at": current_time
            }
            logger.info(f"更新 {mode.upper()} 模式计时: {time_seconds:.2f}秒")
    
    total_time_seconds = sum(
        timing_data["modes"].get(mode, {}).get("dataset_total_time_seconds", 0)
        for mode in ["direct", "cot"]
    )
    
    modes_count = len([mode for mode in ["direct", "cot"] if mode in timing_data["modes"]])
    
    direct_time = timing_data["modes"].get("direct", {}).get("dataset_total_time_seconds", 0)
    cot_time = timing_data["modes"].get("cot", {}).get("dataset_total_time_seconds", 0)
    time_ratio = round(cot_time / direct_time, 2) if direct_time > 0 else 0
    
    timing_data["summary"] = {
        "total_time_seconds": round(total_time_seconds, 2),
        "total_time_minutes": round(total_time_seconds / 60, 2),
        "total_time_hours": round(total_time_seconds / 3600, 2),
        "modes_count": modes_count,
        "updated_at": current_time,
        "time_ratio": {
            "description": "cot/direct",
            "ratio": time_ratio
        }
    }
    
    timing_data["last_updated"] = current_time
    
    # --- 关键修复：在写入文件前，确保其所在的目录存在 ---
    logger.info(f"正在确保目录 '{os.path.dirname(timing_file)}' 存在...")
    os.makedirs(os.path.dirname(timing_file), exist_ok=True)
    
    with open(timing_file, 'w', encoding='utf-8') as f:
        json.dump(timing_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"计时结果已成功保存到: {timing_file}")
    logger.info(f"当前总推理时间: {total_time_seconds:.2f}秒 ({total_time_seconds/60:.2f}分钟)")
    if modes_count == 2:
        logger.info(f"CoT/Direct 时间比例: {time_ratio}")

def main():
    """
    主执行函数，用于调用修复逻辑
    """
    logger.info("--- 开始执行计时结果修复脚本 ---")

    # --- 步骤一：填入从日志中提取的数据 ---
    output_dir = '/z_data/byl/MME-CoT-benchmarks/Med-CoT-Eval_final/final_output'
    model_name = 'HuatuoGPT-Vision-7B'
    mode_times = {
        'direct': 644.17,
        'cot': 9170.70
    }

    logger.info(f"模型名称: {model_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"模式耗时: {mode_times}")

    try:
        # --- 步骤二：调用修复好的函数 ---
        save_timing_results(model_name, mode_times, output_dir)
        logger.info("\n--- 修复成功！计时文件已生成。 ---")
    except Exception as e:
        logger.error(f"\n--- 修复过程中发生错误: {e} ---")
        # 打印详细的错误追溯信息
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()