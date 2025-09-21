# Preparing Benchmarks！

- [ Preparing Benchmarks！](#model-zoo)
  - [Training Datasets](#training-datasets)
  - [Test Datasets](#test-datasets)
  - [Combine Together](#combine-together)




## Training Datasets
Here, we provide the [7-task setting](#7-task-setting-ours) required for the RAM&RAM++.  
In addition, we also offer the commonly used [3-task setting](#3-task-setting) and [5-task setting](#5-task-setting) in academic research, so you can choose the appropriate setting according to your needs!

### 3-Task Setting
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
    <th> [<a href="https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2">HomePage</a>]</th>
    <th> Dehaze </th>
      </tr>
      <tr>
        <td>Rain-100L</td>
        <th>Train & Test</th>
        <th>[<a href="https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f">Download</a>]</th>
        <th>Derain</th>
      </tr>
      <tr>
        <td>BSD400</td>
        <th>Train</th>
        <th>[<a href="https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing">Download</a>]</th>
        <th>Denoise</th>
      </tr>
      <tr>
        <td>WaterlooED</td>
        <th>Train</th>
        <th>[<a href="https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing">Download</a>]</th>
        <th>Denoise</th>
      </tr>
  </tbody>
</table>





### 5-Task Setting
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
        <td>OTS_BETA </td>
    <th>Train </th>
    <th> [<a href="https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2">HomePage</a>]</th>
    <th> Dehaze </th>
      </tr>
      <tr>
        <td>Rain-100L</td>
        <th>Train & Test</th>
        <th>[<a href="https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f">Download</a>]</th>
        <th>Derain</th>
      </tr>
      <tr>
        <td>BSD400</td>
        <th>Train</th>
        <th>[<a href="https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing">Download</a>]</th>
        <th>Denoise</th>
      </tr>
      <tr>
        <td>WaterlooED</td>
        <th>Train</th>
        <th>[<a href="https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing">Download</a>]</th>
        <th>Denoise</th>
      </tr>
      <tr>
        <td>LOL-v1</td>
        <th>Train & Test</th>
        <th>[<a href="https://daooshee.github.io/BMVC2018website/">HomePage</a>]</th>
        <th>Low Light Enhancement</th>
      </tr>
      <tr>
        <td>GoPro</td>
        <th>Train & Test</th>
        <th>[<a href="https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing">Download</a>]</th>
        <th>Motion Deblur</th>
      </tr>
  </tbody>
</table>



### 7 Task Setting (Ours)
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
      <td>GoPro</td>
      <th>Train & Test</th>
      <th>[<a href="https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing">Download</a>]</th>
      <th>Motion Deblur</th>
    </tr>
    <tr>
       <td>OTS_ALPHA </td>
  <th>Train </th>
  <th> [<a href=https://pan.baidu.com/s/1wBE9wh9nXkvcJ6763CX1TA>Baidu Cloud(f1zz)</a>]</th>
      <th>Dehaze</th>
    </tr>
    <tr>
      <td>Rain-13k</td>
      <th>Train & Test</th>
      <th>[<a href="https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe">Download</a>]</th>
      <th>Derain</th>
    </tr>
    <tr>
      <td>LOL-v2</td>
      <th>Train & Test</th>
      <th>[Real Subset <a href="https://pan.baidu.com/share/init?surl=pQW7zq4yqU1zMRrlotxkXg">Baidu Cloud(65ay)</a>] / [Synthetic Subset <a href="https://pan.baidu.com/share/init?surl=t5OYgDgk3mQO53OXqW7QEA">Baidu Cloud(b14u)</a>]</th>
      <th>Low Light Enhancement</th>
    </tr>
    <tr>
      <td>LSDIR</td>
      <th>Train & Test</th>
      <th>[<a href="https://data.vision.ee.ethz.ch/yawli/index.html">HomePage</a>]</th>
      <th>Denoise DeJPEG DeBlur</th>
    </tr>
  </tbody>
</table>

## Test Datasets
We also provide both  [in-distribution](#in-distribution) and [out-of-distribution](#out-of-distribution)test sets for various restoration tasks, facilitating the evaluation of model performance and generalization ability.


### In-Distribution
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
      <td>SOTS</td>
      <th>Test</th>
      <th>[<a href="https://www.kaggle.com/datasets/balraj98/synthetic-objective-testing-set-sots-reside?resource=download">Download</a>]</th>
      <th>3/5/7-task Dehaze</th>
    </tr>
    <tr>
       <td>Rain-100L</td>
       <th>Test</th>
       <th>[<a href="https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f">Download</a>]</th>
       <th>3/5-task Derain</th>
    </tr>
    <tr>
       <td>Rain13k-val</td>
       <th>Test</th>
       <th>[<a href="https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f">Download</a>]</th>
       <th>7-task Derain</th>
    </tr>
    <tr>
      <td>CBSD68</td>
      <th>Test</th>
      <th>[<a href="https://github.com/clausmichele/CBSD68-dataset/tree/master">Download</a>]</th>
      <th>3/5/7-task Denoise</th>
    </tr>
     <tr>
      <td>GoPro</td>
      <th>Test</th>
      <th>[<a href="https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing">Download</a>]</th>
      <th>5/7-task Motion Deblur</th>
    </tr>
    <tr>
        <td>LOL-v1</td>
        <th>Test</th>
        <th>[<a href="https://daooshee.github.io/BMVC2018website/">HomePage</a>]</th>
        <th>5/7-task Low Light Enhancement</th>
    </tr>
    <tr>
        <td>LSDIR-val</td>
        <th>Test</th>
        <th>[<a href="https://data.vision.ee.ethz.ch/yawli/index.html">HomePage</a>]</th>
        <th>7-task DeJPEG DeBlur</th>
      </tr>
  </tbody>
</table>


### Out-of-Distribution
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
      <td>UIEB</td>
      <th>Test</th>
      <th>[<a href="https://li-chongyi.github.io/proj_benchmark.html">HomePage</a>]</th>
      <th>Underwater Enhancement</th>
    </tr>
    <tr>
      <td>Urban100</td>
      <th>Test</th>
      <th>[<a href="https://drive.google.com/file/d/1687jSIjwMyF8MO9MYOZXT4OKJDS9HArD/view?usp=sharing">Download</a>]</th>
      <th>OOD Denoise</th>
    </tr>
    <tr>
      <td>CDD11-test</td>
      <th>Test</th>
      <th>[<a href="https://1drv.ms/f/s!As3rCDROnrbLgqpezG4sao-u9ddDhw?e=A0REHx">Download</a>]</th>
      <th>OOD 7-task (extreme and mixed)</th>
    </tr>
  </tbody>
</table>

## Combine Together

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
