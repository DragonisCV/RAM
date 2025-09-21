<!-- # <p align=center> :fire: `Restore Anything with Masks：Leveraging Mask Image Modeling for Blind All-in-One Image Restoration (ECCV2024)`</p> -->
<p align="center">
  <img src='.assets/logo.png' alt='RAM_LOGO' width='200px'/><br/>
</p>

## <div align="center"><a href="https://rq-wu.github.io/projects/RAM/index.html">Homepage</a> | <a href="https://arxiv.org/abs/2409.19403v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1CDX02vmpPoeWBahvvg2OAH8jwhtBwwmB?usp=drive_link">Google Drive</a> | <a href="">Baidu Cloud(TBD)</a> | <a href="https://huggingface.co/spaces/233zzl/RAM_plus_plus">DEMO</a> | <a href="https://huggingface.co/233zzl/RAM_plus_plus">HuggingFace</a>


<div align="center">

:newspaper:[**News**](#newspaper-news) | :wrench:[**Install**](#wrench-dependencies-and-installation) | :cd:[**Models Zoo**](#cd-pretrained-models) | :camera:[**Quick Demo**](#camera-quick-demo) | :robot: [**Train&Test**](#robot-training-ramram-from-scratch)
 | :robot:[**Datasets**](docs/benchmarks.md) | :scroll:[**License**](#scroll-license) | :postbox: [**Contact**](#postbox-contact) | :handshake: [**Acknowledgements**](#handshake-acknowledgements)

</div>

This is the official PyTorch codes for the paper.  
>**Restore Anything with Masks：Leveraging Mask Image Modeling for Blind All-in-One Image Restoration**<br>  [Chujie Qin](https://github.com/Dragonisss), [Ruiqi Wu](https://rq-wu.github.io/), [Zikun Liu](), [Xin Lin](https://linxin0.github.io/), [Chunle Guo](https://scholar.google.com/citations?user=RZLYwR0AAAAJ&hl=en), [Hyun Hee Park](s), [Chongyi Li<sup>†</sup>](https://li-chongyi.github.io/)<br/>
> ( † indicates corresponding author )<br/>
> In ECCV 2024, \[[Paper Link](https://arxiv.org/abs/2409.19403v1)\]

> **RAM++: <u>R</u>obust Representation Learning via <u>A</u>daptive <u>M</u>ask for All-in-One Image Restoration**<br>
> [Zilong Zhang<sup>*</sup>](https://github.com/Zilong-Zhang003), [Chujie Qin<sup>*</sup>](https://github.com/DragonisCV), [Chunle Guo](https://mmcheng.net/clguo/), [Yong Zhang](), [Chao Xue](), [Ming-Ming Cheng](https://mmcheng.net/cmm/), [Chongyi Li<sup>†</sup>](https://li-chongyi.github.io/)<br/>
> (<sup>*</sup>indicates equal contribution; <sup>†</sup> indicates corresponding author)<br/>
> arxiv preprint, \[[HomePage](https://zilong-zhang003.github.io/RAM2.0/)\], \[[Paper Link](https://arxiv.org/abs/2509.12039)\]


### :rocket: Highlights:
- RAM is a Blind All-In-One Image Restoration framework that can simultaneously handle <b style='font-size: large'>7 Restoration Tasks</b>  and achieve <b style='font-size: large'>SOTA performance</b> !
- RAM focus on tackling how to extract <b style='font-size: large'>Image Prior</b> instead of degradation prior from diverse corrupted images by Leveraging <b style='font-size: large'>Mask Image Modeling</b>.

<h2 style="background:#ffe4b2; padding:10px; border-radius:8px; display:inline-block;">🌟 RAM++ Highlights 🌟</h2>

<table width="90%" style="background:#f9f9f9;border-radius:10px;border:1px solid #e0e0e0;">
  <tr>
    <td>
      <ul>
        <li><b>More Robust & Balanced & Powerful:</b> Excels across <b>seen, unseen, extreme, and mixed degradations</b>.</li>
        <li><b>Enhanced Mask Pretraining:</b> Improved focus on <b>image semantics</b>.</li>
        <li><b>DINO-v2 Regularizer:</b> Maintains <b>stronger representations</b> during fine-tuning.</li>
      </ul>
    </td>
  </tr>
</table>

## Notice!!!
- A persistent distribution shift between standard AIOIR (and broader low-level vision) training sets and real-world degradations constrains generalization; we plan to mitigate this gap in subsequent work.
- [DINOv3](https://arxiv.org/abs/2508.10104) maintains dense feature maps over long-time training, which benefits image restoration; accordingly, we plan to replace DINOv2 with DINOv3.
- **RAM++** shows minimal performance decay as task count grows, indicating strong scaling potential; we encourage fine-tuning or re-training on larger, real-world datasets.

</div>

## :newspaper: News
<ul>
  <li><b>Sep 20, 2025</b>: Dec 27, 2023: Update an extension version of our ECCV 24 paper (Project Page/Paper).</li>
  <li><b>Feb 24, 2025</b>: A Jittor Version is available at <a href="https://github.com/Dragonisss/RAM-Jittor">RAM-Jittor</a>.</li>
  <li><b>Oct 20, 2024</b>: Release pretrained weights on <a href="https://drive.google.com/drive/folders/1CDX02vmpPoeWBahvvg2OAH8jwhtBwwmB?usp=drive_link">Google Drive</a>.</li>
  <li><b>Oct 3, 2024</b>: Release related code of our paper.</li>
</ul>


## :wrench: Dependencies and Installation
1. Clone and enter our repository:
    ```bash
   git clone https://github.com/Dragonisss/RAM.git RAM
   cd RAM
    ```
2. Simply run the `install.sh` for installation!
    ```sh
    source install.sh
    ```
3. Activate the environment whenever you test!
    ```bash
    conda activate RAM
    ```

## :cd: Pretrained Models
> If your requirement is for **academic research** and you would like to benchmark our method, please refer to [pretrained_models.md](docs/pretrained_models.md), where we have a rich variety of models available across a diverse range training strategies, pre-training, and fine-tuning models.

Our pipeline can be applied to any image restoration network. We provide the pre-trained and fine-tuned model files for SwinIR and PromptIR mentioned in the paper.

> 🌟 **New: RAM++ pretrained and finetuned models are now available!** 🌟
You can download all the mentioned model weights via <strong>[Hugging Face](https://huggingface.co/233zzl/RAM_plus_plus)</strong> or simply download only the models you are interested in!
<table>
<thead>
  <tr>
    <th> Method </th>
    <th> Phase </th>
    <th> Framework </th>
    <th> Download Links </th>
    <th> Config File </th>
  </tr>
</thead>
<tbody>
    <tr style="background-color: #f5f5f5;">
    <td>RAM++ </td>
     <th> Pretrain  </th>
    <th> Restormer </th>
    <th> [<a href="https://drive.google.com/drive/folders/1RmmFoXAVagGrVc3fszbkCLKnj2rFTL5b?usp=drive_link">GoogleDrive</a>] </th>
    <th> [<a href="options/RAM_Plus/7task/7task_pretrain.yaml">options/RAM_Plus/7task/7task_pretrain.yaml]</th>
  </tr>
    <tr style="background-color: #f5f5f5;">
    <td>RAM++ </td>
     <th> Finetune  </th>
    <th> Restormer </th>
    <th> [<a href="https://drive.google.com/drive/folders/1RmmFoXAVagGrVc3fszbkCLKnj2rFTL5b?usp=drive_link">GoogleDrive</a>] </th>
    <th>[<a href="options/RAM_Plus/7task/7task_pretrain.yaml">options/RAM_Plus/7task/7task_finetune.yaml] </th>
  </tr>
  <tr>
    <td>RAM </td>
    <th> Pretrain </th>
    <th> SwinIR </th>
    <th> [<a href="https://drive.google.com/file/d/1MsFZe50V5o-ASVBeCY92F1POfJtbLH_D/view?usp=drive_link">GoogleDrive</a>] </th>
    <th> [<a href="options/RAM_SwinIR/ram_swinir_pretrain.yaml">options/RAM_SwinIR/ram_swinir_pretrain.yaml</a>] </th>
  </tr>
   <tr>
    <td>RAM </td>
    <th> Finetune </th>
    <th> SwinIR </th>
    <th> [<a href="https://drive.google.com/file/d/1IHQ9Yw2ajY8oYTKfZkdOgnSk0iexKNj5/view?usp=drive_link">GoogleDrive</a>] </th>
    <th> [<a href="options/RAM_SwinIR/ram_swinir_finetune.yaml">options/RAM_SwinIR/ram_swinir_finetune.yaml</a>] </th>
  </tr>
    <tr>
    <td>RAM </td>
    <th> Pretrain </th>
    <th> PromptIR </th>
    <th> [<a href="https://drive.google.com/file/d/191nk9er4v00Z1RuW6hRGSKb4LlEF0O8a/view?usp=drive_link">GoogleDrive</a>] </th>
    <th> [<a href="options/RAM_PromptIR/ram_promptir_pretrain.yaml">options/RAM_PromptIR/ram_promptir_pretrain.yaml</a>] </th>
  </tr>
    <tr>
    <td>RAM </td>
    <th> Finetune </th>
    <th> PromptIR </th>
    <th> [<a href="https://drive.google.com/file/d/1cqQoUxMNNVFcsR6lKHdZb-2Se80APlcQ/view?usp=drive_link">GoogleDrive</a>] </th>
    <th> [<a href="options/RAM_PromptIR/ram_promptir_finetune.yaml">options/RAM_PromptIR/ram_promptir_finetune.yaml</a>] </th>
  </tr>
</tbody>
</table>

To build RAM++, please download below weights and [**DINOv2**](https://huggingface.co/facebook/dinov2-giant) **FIRST**, and place them under the `./pretrained_model` .


## :camera: Quick Demo
We provide scripts for inference your own images in [inference/inference.py](inference/inference.py). <br/>
You could run `python inference/inference.py --help` to get detailed information of this scripts.

## :robot: Training RAM/RAM++ From Scratch!
Before proceeding, please **ensure** that the relevant datasets have been prepared as required. You can download required datasets by following [benchmarks.md](docs/benchmarks.md)

**1.Pretraining with MIM**
We use the collected datasets for model training. First, we execute the following command:
```python
torchrun \
--nproc_per_node=[num of gpus] \
--master_port=[PORT] ram/train.py \
-opt [OPT] \
--launcher pytorch

# e.g.
torchrun \
--nproc_per_node=8 \
--master_port=4321 ram/train.py \
-opt options/RAM_SwinIR/ram_swinir_pretrain.yaml \
--launcher pytorch
```

**2.Mask Attribute Conductance Analysis**

We use proposed Mask Attribute Conductance Analysis to analyze the importance of different layers for finetuning. You can run the following command to conduct MAC analysis:

```python
#============ MAC Analysis For RAM    ============#
python scripts/mac_analysis.py -opt [OPT]
# e.g.
python scripts/mac_analysis.py \
-opt options/RAM_SwinIR/ram_swinir_mac.yml

#============ MAC Analysis For RAM++  ============#
python scripts/adaSAM_mac_analysis.py -opt [OPT]
# e.g.
python scripts/adaSAM_mac_analysis.py \
-opt options/RAM_Plus/3task/3task_mac.yaml

```
For convenience, we have provided the analysis results of tRAM-SwinIR,RAM-PromptIR and RAM++, mentioned in the paper. You can find them in [./mac_analysis_result/](./mac_analysis_result/)

**3.Finetuning**
```python
torchrun \
--nproc_per_node=[num of gpus] \
--master_port=[PORT] ram/train.py \
-opt [OPT] \
--launcher pytorch

# e.g.
torchrun \
--nproc_per_node=8 \
--master_port=4321 ram/train.py \
-opt options/RAM_SwinIR/ram_swinir_finetune.yaml \
--launcher pytorch
```
You can also add `CUDA_DEVICE_VISIBLE=` to choose gpu you want to use.


## :chart_with_upwards_trend: Evaluation 
We have provided a script for fast evaluation:
```python
torchrun \
--nproc_per_node=1 \
--master_port=[PORT] ram/test.py \
-opt [OPT] --launcher pytorch
```
To benchmark the performance of RAM on the test dataset, you can run the following command:
```python
# RAM-SwinIR
torchrun \
--nproc_per_node=1 \
--master_port=4321 ram/test.py \
-opt options/test/ram_swinir_benchmark.yml --launcher pytorch

# RAM-PromptIR
torchrun \
--nproc_per_node=1 \
--master_port=4321 ram/test.py \
-opt options/test/ram_promptir_benchmark.yml --launcher pytorch
```

To benchmark the performance of RAM++ on the test dataset, you can run the following command:
```python
# 3-task
torchrun \
--nproc_per_node=1 \
--master_port=4321 ram/test.py \
-opt options/3task/3task_benchmark.yaml --launcher pytorch

# 5-task
torchrun \
--nproc_per_node=1 \
--master_port=4321 ram/test.py \
-opt options/5task/5task_benchmark.yaml --launcher pytorch

# 7-task
torchrun \
--nproc_per_node=1 \
--master_port=4321 ram/test.py \
-opt options/7task/7task_benchmark.yaml --launcher pytorch
```

## :scroll: License

This code is licensed under the [Pi-Lab License 1.0](LICENSE) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.



## :book: Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{qin2024restore,
  title={Restore Anything with Masks: Leveraging Mask Image Modeling for Blind All-in-One Image Restoration},
  author={Qin, Chu-Jie and Wu, Rui-Qi and Liu, Zikun and Lin, Xin and Guo, Chun-Le and Park, Hyun Hee and Li, Chongyi},
  booktitle={European Conference on Computer Vision},
  pages={364--380},
  year={2024},
  organization={Springer}
}

@misc{zhang2025ramrobustrepresentationlearning,
      title={RAM++: Robust Representation Learning via Adaptive Mask for All-in-One Image Restoration}, 
      author={Zilong Zhang and Chujie Qin and Chunle Guo and Yong Zhang and Chao Xue and Ming-Ming Cheng and Chongyi Li},
      year={2025},
      eprint={2509.12039},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.12039}, 
}
```

## :postbox: Contact

For technical questions, please contact `chujie.qin[AT]mail.nankai.edu.cn` and `zhangzilong[AT]mail.nankai.edu.cn`


## :handshake: Acknowledgements
This work builds based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Some code are borrows from [Restormer](https://github.com/swz30/Restormer) and [AdaMAE](https://github.com/wgcban/adamae). We are grateful to its authors and contributors for their outstanding open-source efforts and support.

We also thank all of our contributors.

<a href="https://github.com/DragonisCV/RAM">
  <img src="https://contrib.rocks/image?repo=DragonisCV/RAM" />
</a>