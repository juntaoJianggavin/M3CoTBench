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
- [Citation](#citation)  
- [Contact](#contact)  

<a name="pipeline"></a>

# :movie_camera: Data Pipeline

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
```

### 2. Downloads the M3CoTBench Database
<a name="2-downloads-the-m3cotbench-database"></a>
This section provides access to the [M3CoTBench Database](https://huggingface.co/datasets/APRIL-AIGC/M3CoTBench), which contains the complete `.png` image data of M3CoTBench and a `.xlsx` file (the file provides the question, answer and annotated CoT steps in the [M3CoTBench Database](https://huggingface.co/datasets/APRIL-AIGC/M3CoTBench)).
🥰You can download [M3CoTBench Database](https://huggingface.co/datasets/APRIL-AIGC/M3CoTBench) to your local path using the following command:

```
huggingface-cli download --repo-type dataset --resume-download APRIL-AIGC/M3CoTBench --local-dir $YOUR_LOCAL_PATH
```

<a name="usage"></a>

# :muscle:Usage

<a name="citation"></a>

<a name="experiments"></a>

# :bar_chart:Experiments

### **Performance score of different methods**


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
