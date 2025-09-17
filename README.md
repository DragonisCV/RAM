# <p align="center"> :fire: <code>RAM++: <u>R</u>obust Representation Learning via <u>A</u>daptive <u>M</u>ask for All-in-One Image Restoration</code></p>

## <div align="center"><a href="https://zilong-zhang003.github.io/RAM2.0/">Homepage</a> | <a href="https://arxiv.org/abs/2509.12039">Paper</a> | <a href="">Google Drive(TBD)</a> | <a href="">Baidu Cloud(TBD)</a>

This is the official PyTorch codes for the extended paper. The base conference version ([RAM](https://arxiv.org/pdf/2409.19403), ECCV 2024) is available [here](https://github.com/DragonisCV/RAM) 
> **RAM++: <u>R</u>obust Representation Learning via <u>A</u>daptive <u>M</u>ask for All-in-One Image Restoration**<br>
> [Zilong Zhang<sup>*</sup>](https://github.com/Zilong-Zhang003), [Chujie Qin<sup>*</sup>](https://github.com/DragonisCV), [Chunle Guo](https://mmcheng.net/clguo/), [Yong Zhang](), [Chao Xue](), [Ming-Ming Cheng](https://mmcheng.net/cmm/), [Chongyi Li<sup>†</sup>](https://li-chongyi.github.io/)<br>
> <sup>*</sup>indicates equal contribution; <sup>†</sup> indicates corresponding author

![framework_img](.assets/pipeline.png)

### :rocket: Highlights:
- RAM++ is a Blind All-In-One Image Restoration framework that achieves  <b style='font-size: large'>Robust, Well-balanced, SOTA </b> performance across seen, unseen, extreme, and mixed degradations.
- RAM++ focus on tackling how to extract <b style='font-size: large'>Image Prior</b> instead of degradation prior from diverse corrupted images by Leveraging <b style='font-size: large'>Adaptive Mask Image Modeling</b>.

## :newspaper: News
<ul>
  <!-- <li><b>Feb 24, 2025</b>: A Jittor Version is available at <a href="https://github.com/Dragonisss/RAM-Jittor">RAM-Jittor</a>.</li>
   <li><b>Oct 20, 2024</b>: Release  on <a href="https://drive.google.com/drive/folders/1CDX02vmpPoeWBahvvg2OAH8jwhtBwwmB?usp=drive_link">Google Drive</a>.</li> -->
  <li><b>Sep 17, 2025</b>: Release related code of our paper.</li>
</ul>

## :tada: TBD
<li>Release related checkpoints of our paper.</li>
<li>Release 3-task datasets of our paper.</li>


## :wrench: Dependencies and Installation
1. Clone and enter our repository:
    ```bash
   git clone https://github.com/DragonisCV/RAM.git RAM2.0
   cd RAM2.0
    ```
2. Simply run the `install.sh` for installation!
    ```sh
    source install.sh
    ```
3. Activate the environment whenever you test!
    ```bash
    conda activate RAM2
    ```
## :sparkles: Datasets and Pretrained Models
> Given the number of datasets involved, we plan to offer a unified download link in the future to make it easier to access all datasets.

We combine datasets from various restoration tasks to form the training set. Here are the relevant links for all the datasets used:
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Phase</th>
        <th>Source</th>
        <th>Task for</th>
      </tr>
    </thead>
    <tbody>
    <tr>
        <td>OTS_BETA </td>
    <th>Train </th>
    <th> [<a href="">Coming Soon</a>]</th>
    <th> 3-task Dehaze </th>
      </tr>
      <tr>
        <td>Rain-100L</td>
        <th>Train & Test</th>
        <th>[<a href="">Coming Soon</a>]</th>
        <th>3/5-task Derain</th>
      </tr>
      <tr>
        <td>BSD400</td>
        <th>Train</th>
        <th>[<a href="">Coming Soon</a>]</th>
        <th>3/5-task Denoise</th>
      </tr>
      <tr>
        <td>WaterlooED</td>
        <th>Train</th>
        <th>[<a href="">Coming Soon</a>]</th>
        <th>3/5-task Denoise</th>
      </tr>
      <tr>
        <td>LOL-v1</td>
        <th>Train & Test</th>
        <th>[<a href="">Coming Soon</a>]</th>
        <th>5-task Low Light Enhancement</th>
      </tr>
      <tr>
        <td>GoPro</td>
        <th>Train & Test</th>
        <th>[<a href="https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing">Download</a>]</th>
        <th>5/7-task Motion Deblur</th>
      </tr>
      <tr>
         <td>OTS_ALPHA </td>
    <th>Train </th>
    <th> [<a href=https://pan.baidu.com/s/1wBE9wh9nXkvcJ6763CX1TA>Baidu Cloud(f1zz)</a>]</th>
        <th>7-task Dehaze</th>
      </tr>
      <tr>
        <td>Rain-13k</td>
        <th>Train & Test</th>
        <th>[<a href="https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe">Google Drive</a>]</th>
        <th>7-task Derain</th>
      </tr>
      <tr>
        <td>LOL-v2</td>
        <th>Train & Test</th>
        <th>[Real Subset <a href="https://pan.baidu.com/share/init?surl=pQW7zq4yqU1zMRrlotxkXg">Baidu Cloud(65ay)</a>] / [Synthetic Subset <a href="https://pan.baidu.com/share/init?surl=t5OYgDgk3mQO53OXqW7QEA">Baidu Cloud(b14u)</a>]</th>
        <th>7-task Low Light Enhancement</th>
      </tr>
      <tr>
        <td>LSDIR</td>
        <th>Train & Test</th>
        <th>[<a href="https://data.vision.ee.ethz.ch/yawli/index.html">HomePage</a>]</th>
        <th>7-task Denoise DeJPEG DeBlur</th>
      </tr>
      <tr>
        <td>SOTS</td>
        <th>Test</th>
        <th>[<a href="https://www.kaggle.com/datasets/balraj98/synthetic-objective-testing-set-sots-reside?resource=download">Download</a>]</th>
        <th>3/5/7-task Denoise DeJPEG DeBlur</th>
      </tr>
      <tr>
        <td>CBSD68</td>
        <th>Test</th>
        <th>[<a href="https://github.com/clausmichele/CBSD68-dataset/tree/master">Download</a>]</th>
        <th>3/5/7-task Denoise</th>
      </tr>
      <tr>
        <td>UIEB</td>
        <th>Test</th>
        <th>[<a href="https://li-chongyi.github.io/proj_benchmark.html">HomePage</a>]</th>
        <th>OOD 5-task Underwater Enhancement</th>
      </tr>
      <tr>
        <td>Urban100</td>
        <th>Test</th>
        <th>[<a href="">Coming Soon</a>]</th>
        <th>OOD 7-task Denoise</th>
      </tr>
      <tr>
        <td>CDD11-test</td>
        <th>Test</th>
        <th>[<a href="https://1drv.ms/f/s!As3rCDROnrbLgqpezG4sao-u9ddDhw?e=A0REHx">Download</a>]</th>
        <th>OOD 7-task (extreme and mixed)</th>
      </tr>
    </tbody>
  </table>

You need to collect required datasets above and place them under the `./datasets` Directory.

**Symbolic links** is a recommended approach, allowing you to place the datasets anywhere you prefer!

The final directory structure will be arranged as:
```
datasets	              datasets
  |- BSD400	                |- OTS_ALPHA
    |- 2018.jpg	              |- clear
    |- 2092.jpg	              |- depth
    |- ...	                  |- haze
  |- CBSD68	                |- OTS_BETA
    |- CBSD68	              |- clear
      |- noisy5	              |- depth
      |- noisy10	          |- haze
      |- ...	            |- Rain100L
  |- CDD-11_test	          |- norain-1.png
    |- clear	              |- rain-1.png
    |- haze	                  |- ...
    |- ...	                |- rain13k
  |- gopro	                  |- test
    |- test	                  |- train
    |- train                |- SOTS
  |- LOL	                  |- outdoor
    |- test	                |- UIEB
    |- train	              |- raw-890
  |- LOL-v2	                  |- reference-890
    |- Real_captured        |- urban100
    |- Synthetic	          |- urban100_pepper
  |- LSDIR	                  |- urban100_speckle
    |- 0001000	              |- ...
    |- 0002000	            |- WaterlooED
    |- ...	                  |- WaterlooED
  |- LSDIR-val	                 |- 00001.bmp
    |- 0000001.png	             |- ...
    |- 0000002.png	
    |- ...	

```

We provide the pre-trained and fine-tuned model files in three different settings mentioned in the paper. Please download below weights and [DINOv2](https://huggingface.co/facebook/dinov2-giant), and place them under the `./pretrained_model` Directory.
<table>
<thead>
  <tr>
    <th> Method </th>
    <th> Count </th>
    <th> Phase </th>
    <th> Download Links </th>
    <th> Config File </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>RAM++ </td>
     <th> 3-task </th>
    <th> Pretrain </th>
    <th> [<a href="">Coming Soon</a>] </th>
    <th> options/3task/3task_pretrain.yaml </th>
  </tr>
   <tr>
    <td>RAM++ </td>
     <th> 3-task  </th>
    <th> Finetune </th>
    <th> [<a href="">Coming Soon</a>] </th>
    <th> options/3task/3task_finetune.yaml </th>
  </tr>
    <tr>
    <td>RAM++ </td>
     <th> 5-task  </th>
    <th> Pretrain </th>
    <th> [<a href="">Coming Soon</a>] </th>
    <th> options/5task/5task_pretrain.yaml </th>
  </tr>
    <tr>
    <td>RAM++ </td>
     <th> 5-task  </th>
    <th> Finetune </th>
    <th> [<a href="">Coming Soon</a>] </th>
    <th> options/5task/5task_finetune.yaml </th>
  </tr>
  <tr>
    <td>RAM++ </td>
     <th> 7-task  </th>
    <th> Pretrain </th>
    <th> [<a href="">Coming Soon</a>] </th>
    <th> options/7task/7task_pretrain.yaml </th>
  </tr>
    <tr>
    <td>RAM++ </td>
     <th> 7-task  </th>
    <th> Finetune </th>
    <th> [<a href="">Coming Soon</a>] </th>
    <th>options/7task/7task_finetune.yaml </th>
  </tr>
</tbody>
</table>

## :camera: Quick Demo
We provide scripts for inference your own images in [inference/inference.py](inference/inference.py). <br/>
You could run `python inference/inference.py --help` to get detailed information of this scripts.

## :robot: Training RAM++ From Scratch!
Before proceeding, please **ensure** that the relevant datasets have been prepared as required.

**1.Pretraining with AdaSAM**
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
-opt options/3task/3task_pretrain.yaml \
--launcher pytorch
```

**2.Mask Attribute Conductance Analysis**

We use proposed Mask Attribute Conductance Analysis to analyze the importance of different layers for finetuning. You can run the following command to conduct MAC analysis:
```python
python scripts/adaSAM_mac_analysis.py -opt [OPT] --launcher pytorch

# e.g.
python scripts/adaSAM_mac_analysis.py \
-opt options/3task/3task_mac.yaml --launcher pytorch

```
For convenience, we have provided the analysis results of the two models, 3-task and 7-task, mentioned in the paper. You can find them in [./mac_analysis_result/](./mac_analysis_result/)

**3.Finetuning with RFR**
```python
torchrun \
--nproc_per_node=<num of gpus> \
--master_port=4321 ram/train.py \
-opt [OPT] \
--launcher pytorch

# e.g.
torchrun \
--nproc_per_node=8 \
--master_port=4321 ram/train.py \
-opt options/3task/3task_finetune.yaml \
--launcher pytorch
```
You can switch configuration files, e.g., [options/7task/7task_ratio0.3_finetune.yaml](options/7task/7task_ratio0.3_finetune.yaml), to finetune models with different fine-tuning ratios.\
You can also add `CUDA_DEVICE_VISIBLE=` to choose gpu you want to use.


## :chart_with_upwards_trend: Evaluation 
We have provided a script for fast evaluation:
```python
torchrun \
--nproc_per_node=1 \
--master_port=[PORT] ram/test.py \
-opt [OPT] --launcher pytorch
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
You can switch configuration files, e.g., [options/7task/7task_ratio0.3_benchmark.yaml](options/7task/7task_ratio0.3_benchmark.yaml), to test models with different fine-tuning ratios.\
You can also add `OMP_NUM_THREADS= & MKL_NUM_THREADS=` to avoid CPU bottlenecks.
## :book: Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
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

## :handshake: Acknowledgements

This work builds upon the [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. We are grateful to its authors and contributors for their outstanding open-source efforts and support.


## :postbox: Contact

For technical questions, please contact `zhangzilong[AT]mail.nankai.edu.cn`