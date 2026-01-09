<p align="center">
    <a href="">
<img width="500" alt="image" src="assets/title.png">
     </a>
   <p align="center">

<p align="center">
    <a href="#"><strong>Juntao Jiang <sup>1★</sup></strong></a>
    ·
    <a href="https://zhangzjn.github.io/"><strong>Jiangning Zhang <sup>1★</sup></strong></a>
    ·
    <a href="#"><strong>Yali Bi <sup>2</sup></strong></a>
    ·
    <a href="#"><strong>Jinsheng Bai <sup>1</sup></strong></a>
    ·
    <a href="#"><strong>Weixuan Liu <sup>3</sup></strong></a>
    ·
    <a href="#"><strong>Weiwei Jin <sup>4</sup></strong></a>
    ·
    <a href="https://xzc-zju.github.io/"><strong>Zhucun Xue <sup>1</sup></strong></a>
    ·
    <a href="https://person.zju.edu.cn/yongliu"><strong>Yong Liu <sup>1†</sup></strong></a>
    ·
    <a href="https://huuxiaobin.github.io/"><strong>Xiaobin Hu <sup>2</sup></strong></a>
    ·
    <a href="https://yanshuicheng.info/"><strong>Shuicheng Yan <sup>5</sup></strong></a>
</p>

<p align="center">
    <strong><sup>1</sup>Zhejiang University</strong> &nbsp;&nbsp;&nbsp;
    <strong><sup>2</sup>University of Science and Technology of China</strong> &nbsp;&nbsp;&nbsp;
    <strong><sup>3</sup>East China Normal University</strong>
    <br>
    <strong><sup>4</sup>Zhejiang Provincial People’s Hospital</strong> &nbsp;&nbsp;&nbsp;
    <strong><sup>5</sup>National University of Singapore</strong>
</p>

<p align="center">
    <a href=''>
      <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='arXiv PDF'>
         </a>
<a href="https://huggingface.co/datasets/APRIL-AIGC/M3CoTBench"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Dataset&color=yellow"></a>
    <a href='https://juntaojianggavin.github.io/projects/M3CoTBench'>
      <img src='https://img.shields.io/badge/IVEBench-Website-green?style=flat&logo=googlechrome&logoColor=green' alt='webpage-Web'>
         </a>
</p>
<a name="introduction"></a>

# :blush:Continuous Updates

This repository is a comprehensive collection of resources for **M3CoTBench**, If you find any work missing or have any suggestions, feel free to pull requests or [contact us](#contact). We will promptly add the missing papers to this repository.

<a name="highlight"></a>
# ✨ Highlight!!!

Compared with existing multimodal medical benchmarks, our proposed **M3CoTBench** offers the following key advantages:

1. **Diverse Medical VQA Dataset.**  
   We curate a *1,079-image* medical visual question answering (VQA) dataset spanning *24 imaging modalities*, stratified by difficulty and annotated with *step-by-step reasoning* aligned with real clinical diagnostic workflows.
2. **Multidimensional CoT-Centric Evaluation Metrics.**  
   We propose a comprehensive evaluation protocol that measures *reasoning correctness, efficiency, impact, and consistency*, enabling fine-grained and interpretable analysis of CoT behaviors across diverse MLLMs.
3. **Comprehensive Model Analysis and Case Studies.**  
   We benchmark both general-purpose and medical-domain MLLMs using quantitative metrics and in-depth qualitative case studies, revealing strengths and failure modes in clinical reasoning to guide future model design.



**🤓 You can view the scores and comparisons of each method at [M3CoTBench LeaderBoard](https://juntaojianggavin.github.io/projects/M3CoTBench/#leaderboard).**

# :mailbox_with_mail:Summary of Contents

- [Introduction](#introduction)  
- [Highlight](#highlight)  
- [Data Pipeline](#pipeline)  
- [Benchmark Overview](#benchmark-overview)  
- [Installation](#installation)  
  - [Install requirements](#1-install-requirements)  
  - [Download M3CoTBench Database](#2-downloads-the-m3cotbench-database)  
- [Usage](#usage)  
- [Experiments](#experiments)
- [Acknowlegments](#acknowledgments)  
- [Citation](#citation)  
- [Contact](#contact)  

<a name="pipeline"></a>

# :microscope: Data Pipeline

<img src="assets/curation.png" width.="1000px">

**Data acquisition and annotation pipeline of M3CoTBench.** **a)** Carefully curated medical images from various public sources. **b)** Multi-type and multi-difficulty QA generation via LLMs and expert calibration. **c)** Structured annotation of key reasoning steps aligned with clinical diagnostic workflows.



<a name="benchmark-overview"></a>

# :sunflower: Benchmark Overview

<img src="assets/overview.png" width.="1000px">

**Overview of M3CoTBench.** **Top:** The benchmark covers 24 imaging modalities/examination types, 4 question types, and 13 clinical reasoning tasks. **Middle:** CoT annotation examples and 4 evaluation dimensions. **Bottom:** The distribution of image-QA pairs across **a)** modalities, **b)** question types, and **c)** tasks.



<a name="installation"></a>

# :hammer:Installation

### 1. Install requirements
<a name="1-install-requirements"></a>

```
git clone https://github.com/juntaoJianggavin/M3CoTBench.git
cd M3CoTBench
```

### 2. Downloads the M3CoTBench Database
<a name="2-downloads-the-m3cotbench-database"></a>
This section provides access to the [M3CoTBench Database](https://huggingface.co/datasets/APRIL-AIGC/M3CoTBench), which contains the complete `.png` image data of M3CoTBench and a `.xlsx` file (the file provides the question, answer and annotated CoT steps in the [M3CoTBench Database](https://huggingface.co/datasets/APRIL-AIGC/M3CoTBench)).
🥰You can download [M3CoTBench Database](https://huggingface.co/datasets/APRIL-AIGC/M3CoTBench) to your local path using the following command:

```
huggingface-cli download --repo-type dataset --resume-download APRIL-AIGC/M3CoTBench --local-dir $YOUR_LOCAL_PATH
```

Then put the M3CoTBench.xlsx and images/ into the M3CoTBench/inference/datasets/


<a name="usage"></a>

# :muscle:Usage
### 3. Inference
#### Specialized Medical Models
Note: Model weights should be placed in M3CoTBench/inference/pretrain. Each medical model requires its own specific Conda environment.
For HealthGPT:
```
conda activate M3CoTBench_healthgpt

bash  M3CoTBench/inference/HealthGPT/llava/demo/run_batch_eval.sh
```
For HuatuoGPT-Vision:
```
conda activate M3CoTBench_huatuo
cd M3CoTBench/inference/pretrain/HuatuoGPT-Vision

python eval_new.py
```
For LLaVA-Med:
```
conda activate M3CoTBench_llava
cd M3CoTBench/inference/pretrain/LLaVA-Med

# Direct Inference
python llava/eval/model_vqa.py --mode direct

# Chain-of-Thought (CoT) Inference
python llava/eval/model_vqa.py --mode cot
```

Note: Lingshu and MedGemma are integrated into the General Framework below.

#### General Framework
Environment: M3CoTBench 

Working Directory: All scripts must be run from M3CoTBench/inference/.

(1) API Inference
```
# Start "GPT-5" on port xxxxx with 4 internal processes
bash run_api_model.sh "GPT-5" xxxxx 4

# Start "Claude-Sonnet-4.5" on port xxxxx (default 4 processes)
bash run_api_model.sh "Claude-Sonnet-4.5" xxxxx
```
(2) Local Inference
```
bash scripts/run_local_gpu_model.sh LLaVA-CoT 1,2,3,4,5,6 all xxxxx
```

To rerun failed inference data and update results:
```
cd M3CoTBench/inference/

# 1. Rerun failed files and merge into the original JSON
python reprocess_failed.py \
    --input-file final_output/Lingshu-32B/Lingshu-32B_direct.json \
    --model "Lingshu-32B" \
    --data-path "dataset/M3CoTBench.xlsx" \
    --image-dir "dataset/images" \
    --update-in-place

# 2. Recalculate timing summary
python recalculate_summary.py \
    --results-file final_output/Lingshu-32B/Lingshu-32B_direct.json \
    --summary-file final_output/Lingshu-32B/Lingshu-32B_summary.json
```

### 4. Evaluation

#### Correctness
Step 1: Merge Chain-of-Thought Fields Merge the CoT steps of the correct answers and convert the format to XLSX.
```
cd M3CoTBench/evaluation/
python combine_fields.py
```
Step 2: Format Inference Results Batch format the inference JSON files into the evaluation output format (XLSX). This file will contain both the CoT of the correct answer and the predicted answer from the inference.
```
python tools/update_lmmseval_json.py
```

Step 3. Run Evaluation Scripts
You can run metrics individually. For example, to evaluate recall:
```
bash scripts/recall.sh
bash scripts/precision.sh
Note: Simply update the data path for YOUR_MODEL_NAME inside recall.sh (or other script files).
```
Alternatively, you can run all metrics for all models in a specific directory using the following command:

```
bash batch_scripts/run_all.py --result_dir results/
```

After the GPT evaluation, you should see a cache/ directory structured as follows:
```
📂 cache
 ┣━━ 📂 recall
 ┃    ┗━━ 📂 YOUR_MODEL_NAME
 ┃         ┣━━ 📄 1.json
 ┃         ┣━━ 📄 2.json
 ┃         ┗━━ 📄 ...
 ┗━━  📂 precision
    ┗━━ 📂 YOUR_MODEL_NAME
```

Step 4. Calculate Metrics
We cache the evaluation results for all questions in the cache directory. Here, we read results from the cache to calculate the final metrics.

For example, to calculate Quality:

```
python final_score/quality.py --cache_dir cache --save_path final_results
```

The script will automatically calculate Recall and Precision, and then compute the F1 Score or Average Score.

Alternatively, you can calculate each metric individually. For example, to calculate Recall:

```
python final_score/recall.py --cache_dir cache/recall --save_path final_results
```


<a name="experiments"></a>

# :bar_chart:Experiments

### **Performance score of different methods.** 
<p><strong>Metrics:</strong> ↑ Higher is Better, ↓ Lower is Better. <strong>Bold</strong>: Best result.</p>

<table>
    <thead>
        <tr>
            <th rowspan="2" align="center"><strong>#</strong></th>
            <th rowspan="2" align="left"><strong>Model</strong></th>
            <th rowspan="2" align="center"><strong>Category</strong></th>
            <th colspan="3" align="center"><strong>Correctness (↑)</strong></th>
            <th colspan="3" align="center"><strong>Impact (↑)</strong></th>
            <th colspan="2" align="center"><strong>Efficiency</strong></th>
            <th rowspan="2" align="center"><strong>Consistency<br>C<sub>path</sub> (↑)</strong></th>
        </tr>
        <tr>
            <th align="center"><strong>F1</strong></th>
            <th align="center"><strong>P</strong></th>
            <th align="center"><strong>R</strong></th>
            <th align="center"><strong>Acc<sub>dir</sub></strong></th>
            <th align="center"><strong>Acc<sub>step</sub></strong></th>
            <th align="center"><strong>I</strong></th>
            <th align="center"><strong>E (↑)</strong></th>
            <th align="center"><strong>L (↓)</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">1</td>
            <td align="left">LLava-CoT</td>
            <td align="center">Open-source</td>
            <td align="center">49.80</td><td align="center">54.08</td><td align="center">46.15</td>
            <td align="center">40.08</td><td align="center">36.75</td><td align="center">-3.33</td>
            <td align="center">0.06</td><td align="center">1.56</td><td align="center">77.02</td>
        </tr>
        <tr>
            <td align="center">2</td>
            <td align="left">InternVL3.5-8B</td>
            <td align="center">Open-source</td>
            <td align="center">56.48</td><td align="center">60.61</td><td align="center">52.88</td>
            <td align="center">56.81</td><td align="center">53.61</td><td align="center">-3.20</td>
            <td align="center">0.10</td><td align="center">18.27</td><td align="center">71.65</td>
        </tr>
        <tr>
            <td align="center">3</td>
            <td align="left">InternVL3.5-30B</td>
            <td align="center">Open-source</td>
            <td align="center">59.42</td><td align="center">62.15</td><td align="center">56.92</td>
            <td align="center"><b>63.81</b></td><td align="center">57.60</td><td align="center">-6.21</td>
            <td align="center">0.03</td><td align="center">16.68</td><td align="center">76.30</td>
        </tr>
        <tr>
            <td align="center">4</td>
            <td align="left">Qwen3-VL-Instruct-8B</td>
            <td align="center">Open-source</td>
            <td align="center">55.17</td><td align="center">52.74</td><td align="center">57.84</td>
            <td align="center">51.30</td><td align="center">46.62</td><td align="center">-4.68</td>
            <td align="center">0.04</td><td align="center">93.94</td><td align="center">82.65</td>
        </tr>
        <tr>
            <td align="center">5</td>
            <td align="left">Qwen3-VL-Instruct-30B</td>
            <td align="center">Open-source</td>
            <td align="center">59.15</td><td align="center">56.13</td><td align="center">62.51</td>
            <td align="center">54.63</td><td align="center">51.39</td><td align="center">-3.24</td>
            <td align="center">0.03</td><td align="center">35.63</td><td align="center"><u>83.01</u></td>
        </tr>
        <tr>
            <td align="center">6</td>
            <td align="left">Qwen3-VL-Thinking-8B</td>
            <td align="center">Open-source</td>
            <td align="center">59.87</td><td align="center">59.84</td><td align="center">59.91</td>
            <td align="center">48.33</td><td align="center">52.83</td><td align="center"><b>+4.50</b></td>
            <td align="center">0.02</td><td align="center">2.79</td><td align="center">76.91</td>
        </tr>
        <tr>
            <td align="center">7</td>
            <td align="left">Qwen3-VL-Thinking-30B</td>
            <td align="center">Open-source</td>
            <td align="center"><u>62.15</u></td><td align="center">63.34</td><td align="center">61.01</td>
            <td align="center">51.90</td><td align="center">55.47</td><td align="center"><u>+3.57</u></td>
            <td align="center">0.02</td><td align="center"><u>1.15</u></td><td align="center">76.02</td>
        </tr>
        <tr>
            <td align="center">8</td>
            <td align="left">GPT-4.1</td>
            <td align="center">Closed-source</td>
            <td align="center">60.76</td><td align="center">58.32</td><td align="center"><u>63.42</u></td>
            <td align="center">56.77</td><td align="center">57.97</td><td align="center">+1.22</td>
            <td align="center">0.17</td><td align="center">5.08</td><td align="center">81.31</td>
        </tr>
        <tr>
            <td align="center">9</td>
            <td align="left">GPT-5</td>
            <td align="center">Closed-source</td>
            <td align="center">55.13</td><td align="center"><u>64.15</u></td><td align="center">48.34</td>
            <td align="center">58.76</td><td align="center">58.29</td><td align="center">-0.47</td>
            <td align="center">0.06</td><td align="center"><b>1.10</b></td><td align="center">65.39</td>
        </tr>
        <tr>
            <td align="center">10</td>
            <td align="left">Gemini 2.5 Pro</td>
            <td align="center">Closed-source</td>
            <td align="center"><b>66.07</b></td><td align="center">62.48</td><td align="center"><b>70.10</b></td>
            <td align="center"><u>60.24</u></td><td align="center"><b>60.06</b></td><td align="center">-0.18</td>
            <td align="center">0.10</td><td align="center">1.52</td><td align="center">82.00</td>
        </tr>
        <tr>
            <td align="center">11</td>
            <td align="left">Claude-Sonnet-4.5</td>
            <td align="center">Closed-source</td>
            <td align="center">56.50</td><td align="center">53.62</td><td align="center">59.71</td>
            <td align="center">51.25</td><td align="center">51.07</td><td align="center">-0.18</td>
            <td align="center">0.15</td><td align="center">2.69</td><td align="center"><b>85.22</b></td>
        </tr>
        <tr>
            <td align="center">12</td>
            <td align="left">LLaVA-Med (7B)</td>
            <td align="center">Medical</td>
            <td align="center">30.51</td><td align="center">36.33</td><td align="center">26.30</td>
            <td align="center">29.38</td><td align="center">29.29</td><td align="center">-0.09</td>
            <td align="center"><b>0.35</b></td><td align="center">3.22</td><td align="center">72.68</td>
        </tr>
        <tr>
            <td align="center">13</td>
            <td align="left">HuatuoGPT-Vision (7B)</td>
            <td align="center">Medical</td>
            <td align="center">49.45</td><td align="center">51.17</td><td align="center">47.85</td>
            <td align="center">41.89</td><td align="center">34.94</td><td align="center">-6.95</td>
            <td align="center">0.21</td><td align="center">5.92</td><td align="center">73.19</td>
        </tr>
        <tr>
            <td align="center">14</td>
            <td align="left">HealthGPT (3.8B)</td>
            <td align="center">Medical</td>
            <td align="center">32.56</td><td align="center">47.27</td><td align="center">24.83</td>
            <td align="center">44.11</td><td align="center">41.98</td><td align="center">-2.13</td>
            <td align="center">0.06</td><td align="center">15.36</td><td align="center">67.72</td>
        </tr>
        <tr>
            <td align="center">15</td>
            <td align="left">Lingshu-7B</td>
            <td align="center">Medical</td>
            <td align="center">57.57</td><td align="center">63.96</td><td align="center">52.34</td>
            <td align="center">50.00</td><td align="center">42.08</td><td align="center">-7.92</td>
            <td align="center"><u>0.30</u></td><td align="center">8.37</td><td align="center">74.83</td>
        </tr>
        <tr>
            <td align="center">16</td>
            <td align="left">Lingshu-32B</td>
            <td align="center">Medical</td>
            <td align="center">59.16</td><td align="center"><b>65.68</b></td><td align="center">53.82</td>
            <td align="center">51.77</td><td align="center">44.95</td><td align="center">-6.82</td>
            <td align="center">0.21</td><td align="center">10.87</td><td align="center">71.47</td>
        </tr>
        <tr>
            <td align="center">17</td>
            <td align="left">MedGemma-4B</td>
            <td align="center">Medical</td>
            <td align="center">48.13</td><td align="center">50.29</td><td align="center">46.14</td>
            <td align="center">43.33</td><td align="center">41.29</td><td align="center">-2.04</td>
            <td align="center">0.05</td><td align="center">20.61</td><td align="center">74.03</td>
        </tr>
        <tr>
            <td align="center">18</td>
            <td align="left">MedGemma-27B</td>
            <td align="center">Medical</td>
            <td align="center">50.98</td><td align="center">48.33</td><td align="center">53.81</td>
            <td align="center">46.06</td><td align="center">45.88</td><td align="center">-0.18</td>
            <td align="center">0.03</td><td align="center">23.71</td><td align="center">82.55</td>
        </tr>
    </tbody>
</table>
<a name="acknowledgments"></a>

# 🙏 Acknowledgments

We would like to acknowledge that some parts of the code were inspired by and referenced from [MME-CoT](https://mmecot.github.io/).


<a name="citation"></a>

# :black_nib:Citation

If you If you find [M3CoTBench](https://juntaojianggavin.github.io/projects/M3CoTBench/) useful for your research, please consider giving a star⭐ and citation📝 :)

```
```



<a name="contact"></a>

# ✉️Contact

```
juntaojiang@zju.edu.cn
```

```
186368@zju.edu.cn
```
