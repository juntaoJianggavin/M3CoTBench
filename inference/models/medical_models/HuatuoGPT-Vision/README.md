# HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale


<div align="center">
<h5>
  📃 <a href="https://arxiv.org/abs/2406.19280" target="_blank">Paper</a>  • 🖥️ <a href="https://vision.huatuogpt.cn/#/" target="_blank">Demo</a>
</h5>
</div>

<div align="center">
<h4>
  📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/PubMedVision" target="_blank">PubMedVision</a> 
</h4>
</div>

<div align="center">
<h4>
  🤗 <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B" target="_blank">HuatuoGPT-Vision-34B</a>  • 🤗 <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B">HuatuoGPT-Vision-7B</a> 
</h4>
</div>

## ✨ Updates
- [04/23/2025]: We release the training code based on the Qwen2.5-VL framework.
- [01/09/2025]: We released the evaluation code (including evaluation data). Additionally, we added instructions on how to train your medical multimodal LLM.  
- [06/28/2024]: We released our medical MLLMs, including [HuatuoGPT-Vision-34B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B) and [HuatuoGPT-Vision-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B).
- [06/26/2024]: We released [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision), a **1.3M** high-quality medical VQA dataset for injecting medical visual knowledge.

## 🩻 PubMedVision
- **PubMedVision** is a large-scale, high-quality medical VQA dataset, constructed from the image-text pairs from PubMed and reformatted using GPT-4V.

|                             | # Data     | Download |
| ------------------ | ---------- | ------------- | 
| **PubMedVision Dataset**   | **1,294,062** | [HF Link](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) |

- **PubMedVision** could significantly improve the medical multimodal capabilities of MLLMs such as LLaVA-v1.5. 

|                                         | **VQA-RAD** | **SLAKE** | **PathVQA** | **PMC-VQA** |
| --------------------------------------- | ----------- | --------- | ----------- | ----------- |
| LLaVA-v1.6-34B                          | 58.6        | 67.3      | 59.1        | 44.4        |
| LLaVA-v1.5-LLaMA3-8B                    | 54.2        | 59.4      | 54.1        | 36.4        |
| LLaVA-v1.5-LLaMA3-8B + **PubMedVision** | **63.8**    | **74.5**  | **59.9**    | **52.7**    |

|                           | **OmniMedVQA**  | **MMMU Health & Medicine (Test Set)** |
| ----------------------------------- | ------------ | -------------------------- |
| LLaVA-v1.6-34B                      | 61.4         | 48.8                       |
| LLaVA-v1.5-LLaMA3-8B                | 48.8         | 38.2                       |
| LLaVA-v1.5-LLaMA3-8B + **PubMedVision** | **75.1**    | **49.1**                   |

## 👨‍⚕️ HuatuoGPT-Vision
HuatuoGPT-Vision is our medical multimodal LLMs, built on **PubMedVision**.

### Model Access
Our model is available on Huggingface in two versions:
|                 | Backbone           | Checkpoint                                                                            |
|----------------------|--------------------|---------------------------------------------------------------------------------------|
| **HuatuoGPT-Vision-7B**  | Qwen2-7B           | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B)             |
| **HuatuoGPT-Vision-34B** | Yi-1.5-34B         | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B)            |

### Model Usage

- **Command Line Interface**

Chat via the command line:
```bash
python cli.py --model_dir path-to-huatuogpt-vision-model
```

- **Model Inference**

Inference using our ChatBot:
```python
query = 'What does the picture show?'
image_paths = ['image_path1']

from cli import HuatuoChatbot
bot = HuatuoChatbot(path-to-huatuogpt-vision-model)
output = bot.inference(query, image_paths)
print(output) # Prints the output of the model
```

### Performance of Medical Multimodal
|                             | **VQA-RAD** | **SLAKE** | **PathVQA** | **PMC-VQA** |
|-----------------------------|-------------|-----------|-------------|-------------|
| LLaVA-Med-7B                   | 51.4        | 48.6      | 56.8        | 24.7        |
| LLaVA-v1.6-34B              | 58.6        | 67.3      | 59.1        | 44.4        |
| **HuatuoGPT-Vision-7B**         | 63.7        | 76.2     | 57.9        | 54.3        |
| **HuatuoGPT-Vision-34B**        | **68.1**    | **76.9**  | **63.5**    | **58.2**    |

|                           | **OmniMedVQA**  | **MMMU Health & Medicine (Test Set)** |
|---------------------------|-----------------|---------------------------------------|
| LLaVA-Med-7B                 | 44.5            | 36.9                                  |
| LLaVA-v1.6-34B            | 61.4            | 48.8                                  |
| **HuatuoGPT-Vision-7B**        | 74.0            | 50.6                                  |
| **HuatuoGPT-Vision-34B**      | **76.9**        | **54.4**                               |

## 📏 Evaluation
1. For evaluation, you need to download the medical evaluation dataset. We have organized multiple evaluation datasets, which can be downloaded directly via the following link:

| Dataset | Link |
| --- | --- |
| Medical_Multimodal_Evaluation_Data | [link](https://huggingface.co/datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data) |  

We have bundled multiple evaluation datasets together. Simply download the data and extract the `images.zip` file.  

2. Then, you can evaluate using the following command:
```bash
accelerate launch eval.py --data_path Medical_Multimodal_Evaluation_Data/medical_multimodel_evaluation_data.json  --model_path HuatuoGPT-Vision-7B
```

3. Once executed, you will directly obtain the results of the medical multimodal evaluation, including datasets like `VQA-RAD`, `SLAKE`, `PathVQA`, `PMC-VQA`, `OmniMedVQA`, and `MMMU-Medical-Tracks`.


## 🏋 Training
This project uses LLaVA's code for training, and it is recommended to use LLaVA's code for training. The code is available at [LLaVA](https://github.com/haotian-liu/LLaVA).  
> To reproduce our results, please train the model using a combination of the PubMedVision dataset and LLaVA's dataset.

> **Update (April 23, 2025):**  
> We release our training process using an newer architecture.  Given that the original LLaVA codebase are relatively outdated, we’ve transitioned to a cleaner and more efficient training framework based on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).  
> The Qwen2.5-VL architecture is more compatible with Huggingface Transformers, making both training and deployment easier, and it offers better support for image processing.  **We strongly recommend using Qwen2.5-VL.** (Note: Use the latest Transformers library for Qwen2.5-VL compatibility.)

- **Step 1: Initialize Qwen2.5-VL with LLMs**

Initialize Qwen2.5-VL with LLMs (Qwen2.5-LLM). This step initializes a Qwen2.5-VL with an LLM so that we can train an MLLM from scratch.

```bash
python convert_llm_to_vl.py \
  --vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --llm_model_path Qwen/Qwen2.5-7B-Instruct \
  --save_path ./Qwen2.5-VL-7B-Base
```

- **Step 2: Vision Alignment (Caption Dataset Pretraining)**

```bash
accelerate launch --config_file ./config/ds.yaml \
  --num_processes 8 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port 29502 \
  --deepspeed_multinode_launcher standard train_vl.py \
  --experiment_name huatuogpt_vision_alignment \
  --run_name huatuogpt_vision \
  --model_path ./Qwen2.5-VL-7B-Base \
  --max_ckpts 1 \
  --gradient_accumulation_steps 8 \
  --data_path Vision_Alignment_Data.json \
  --output_dir ./huatuogpt_vision_alignment_checkpoint \
  --n_epochs 1 \
  --warmup_rates 0.05 \
  --train_bsz_per_gpu 2 \
  --learning_rate 5e-6 \
  --gradient_checkpointing
```

- **Step 3: Vision Instruction Fine-tuning**

```bash
accelerate launch --config_file ./config/ds.yaml \
  --num_processes 8 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port 29502 \
  --deepspeed_multinode_launcher standard train_vl.py \
  --experiment_name huatuogpt_vision_sft \
  --run_name huatuogpt_vision \
  --model_path huatuogpt_vision_alignment_model \
  --max_ckpts 1 \
  --gradient_accumulation_steps 8 \
  --data_path Vision_SFT_Data.json \
  --n_epochs 1 \
  --warmup_rates 0.05 \
  --train_bsz_per_gpu 2 \
  --learning_rate 5e-6 \
  --gradient_checkpointing
```



## 🩺 HuatuoGPT Series 

Explore our HuatuoGPT series:
- [**HuatuoGPT**](https://github.com/FreedomIntelligence/HuatuoGPT): Taming Language Models to Be a Doctor
- [**HuatuoGPT-II**](https://github.com/FreedomIntelligence/HuatuoGPT-II): One-stage Training for Medical Adaptation of LLMs
- [**HuatuoGPT-Vision**](https://github.com/FreedomIntelligence/HuatuoGPT-Vision): Injecting Medical Visual Knowledge into Multimodal LLMs at Scale


## Citation
```
@misc{chen2024huatuogptvisioninjectingmedicalvisual,
      title={HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale}, 
      author={Junying Chen and Ruyi Ouyang and Anningzhe Gao and Shunian Chen and Guiming Hardy Chen and Xidong Wang and Ruifei Zhang and Zhenyang Cai and Ke Ji and Guangjun Yu and Xiang Wan and Benyou Wang},
      year={2024},
      eprint={2406.19280},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.19280}, 
}
```
