# Efficient multiscale computer vision targeting and evidence-based response to crop diseases - Group 4

## Overview

This repository contains the implementation of our research on **Real-time detection of Early Blight (*Alternaria Solani*) in potato crops using Unmanned Aerial Vehicle (UAV) imagery.**

Modern precision agriculture demands a difficult trade-off: deep learning models must be complex enough to generalize across different growing seasons (Domain Generalization) yet lightweight enough to run on drone hardware (Edge Efficiency). To address this, we developed and benchmarked novel **Hybrid CNN-Transformer architectures** that combine the local feature extraction of **ResNet50** with the global context modeling of **CSWin Transformers**.

## Key features of this proejct
* **Novel Hybrid Architectures:** Implementations of both **Sequential** (v2) and **Parallel** (v3) integration strategies to test Feature Refinement vs. Zero-Shot Robustness.
* **Cross-Year Validation:** rigorous testing on UAV dataset (Alternaria solani dataset, Belgium, https://zenodo.org/records/10727413) spanning two distinct agricultural seasons (2019 & 2022) to ensure temporal generalization.
* **Edge Optimization Pipeline:** A complete efficiency benchmark comparing **Knowledge Distillation** (Teacher-Student) against **Iterative Structured Pruning + Dynamic Quantization**.
* **Near-Infrared (NIR) Analysis:** leveraged multispectral sensor data to exploit chlorophyll degradation signatures invisible to standard RGB cameras.


## Dataset
Utilized dataset can be found in the link provided above. Download the zipped file you find there and after unzipping it, you should have 2 folders : `2019` and `2022`. Take those 2 and put them in the `current_data` folder present in the repository. Afterwards, go to `prepare_datasets` folder and run everything from there. These will split the data as needed for all our cases.
<br>

Running `prepare_datasets/get_mean_std_for_normalization.py` should return the following:
* 2019 train entirely:
    * mean: `0.7553`, `0.3109`, `0.1059`
    * std: `0.1774`, `0.1262`, `0.0863`
* 2022 train entirely:
    * mean: `0.7083`, `0.2776`, `0.0762`
    * std: `0.1704`, `0.1296`, `0.0815`


    
### Dataset overview
| Dataset Year | Class Description       | Label | Image Count | Dimensions |
|:-------------|:------------------------|:-----:|:-----------:|:----------:|
| **2019**     | Not *Alternaria Solani* |   0   |    1,710    | 256x256x3  |
|              | *Alternaria Solani*     |   1   |    2,795    | 256x256x3  |
| **2022**     | Not *Alternaria Solani* |   0   |    2,130    | 256x256x3  |
|              | *Alternaria Solani*     |   1   |    1,027    | 256x256x3  |
| **Total**    | **Combined**            | **-** |  **7,662**  |   **-**    |


## Baselines and Hybrid Architectures
Baslines:
1. ResNet50 (DTL) (`phase_performance_baselines/train_deep.py`)
2. VGG11 (DTL) (`phase_performance_baselines/train_deep.py`)

Hybrids:
1. Sequential Model v1 (`phase_performance_hybrids/resnet50_cswin/model_v1.py`)
2. Sequential Model v2 (`phase_performance_hybrids/resnet50_cswin/model_v2.py`)
3. Parallel Model v3 (`phase_performance_hybrids/resnet50_cswin/model_v3.py`)
4. Parallel Model v3 without ResNet50 (`phase_performance_hybrids/resnet50_cswin/model_v3_noresnet.py`)


## Project Structure
- `cross_year_configurations_data` — Cross-year configured data after running scripts from `prepare_datasets`
- `current_data` — Dataset folder
- `phase_efficiency` — Folder containing Jupyter Notebook for Knowledge Distillation, Iterative Structured Pruning and Dynamic Quantization along with their results
- `phase_performance_baselines` — Folder containing the replication of our baselines and their results
- `phase_performance_hybrids` — Folder containing the implementation of all hybrid models, including their training script and results
- `prepare_datasets` — Folder containing script for data splits and calculating mean & standard deviation for them
- `README.md` — README file
<br><br>
In case of finding any folder named `out_of_usage`, please ignore them as they contain all kinds of code for things we did not include in our final work and findings.

## Results Summary
| Experiment / Area          | Key Finding                                                   | Metrics / Comparison                                                              | Implication                                                                                                           |
|:---------------------------|:--------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|
| **Baselines**              | **ResNet50 (DTL)** established as the optimal baseline.       | Matched *AlternarAI* accuracy (~88-91%) but with better F1 balance than VGG11.    | Deep Transfer Learning is sufficient to match current SOTA; Hybrid models must beat this standard.                    |
| **Augmentation**           | **Heavy Augmentation** is essential for Hybrid architectures. | Improved F1-score by **2-3%** compared to Basic Augmentation.                     | Synthetic variance (Random Erasing, Gaussian Blur) forces the Transformer to learn global context over local texture. |
| **Architecture (Forward)** | **Parallel (v3)** excels at Zero-Shot Generalization.         | **91% Accuracy** (v3) vs. 85% (v2) on '19 $\to$ '22 split.                        | Parallel design builds independent global representations, making it more robust to unseen future domains.            |
| **Architecture (Inverse)** | **Sequential (v2)** excels at Feature Refinement.             | **89% Accuracy** (v2) vs. 88% (v3) on '22 $\to$ '19 split.                        | Sequential design polishes CNN features better when training data is abundant/diverse.                                |
| **Ablation Study**         | **Convolutional Stem** is non-negotiable.                     | Removing ResNet caused a catastrophic drop: **91% $\to$ 68%** Accuracy.           | Vision Transformers lack the inductive bias to detect high-frequency lesion boundaries without a CNN stem.            |
| **Efficiency (Forward)**   | **Pruning + Quantization** is superior for robustness.        | Retained **0.92 F1** (vs. 0.91 for Distillation) while reducing size to **35MB**. | For mission-critical tasks, compressing the Hybrid model is better than distilling it to a lightweight student.       |
| **Efficiency (Inverse)**   | **Distillation** is sufficient for data-rich regimes.         | EfficientNet-B0 matched Hybrid performance (**0.86 F1**) at just **15.6MB**.      | When training data is abundant, the complex Hybrid backbone becomes redundant for inference.                          |
| **Data Efficiency**        | Performance gains were incremental, not exponential.          | Hybrid models saturated quickly due to limited data (**~7,600 patches**).         | The "Data Hunger" of Transformers bottlenecked peak performance; larger datasets are needed to unlock full potential. |

## How to run
After following the steps presented under `Dataset`, you should be able to run any file we have in this repository. <br>
Due to GitHub upload limitations, we can't provide model checkpoints, for which we have saved after every experiment and training phase we have performed. In case this is needed, feel free to contact any of us, so we can grant you access to the Google Drive. <br>

## Package requirements
Before running the code, make sure you have the following Python Packages intalled:
1. `torch>=2.4.0`
2. `torchvision.*`
3. `timm`
4. `einops`
5. `numpy`
6. `scikit-learn`

## Authors
**[Horia Ionescu](mailto:h.ionescu@student.maastrichtuniversity.nl), [Dan Loznean](mailto:d.loznean@student.maastrichtuniversity.nl), [Janik Euskirchen](mailto:j.euskirchen@student.maastrichtuniversity.nl), [Vasile Mereuţă](mailto:v.mereuta@student.maastrichtuniversity.nl), [Stan Ostaszewski](mailto:s.ostaszewski@student.maastrichtuniversity.nl), [Gunes Özmen Bakan](mailto:g.ozmen@student.maastrichtuniversity.nl)**  
**Supervisors: [Dr. Charis Kouzinopoulos](mailto:charis.kouzinopoulos@maastrichtuniversity.nl), [Dr. Marcin Pietrasik](mailto:marcin.pietrasik@maastrichtuniversity.nl)** <br>
**Coordinator: [Dr. Gijs Schoenmakers](mailto:gm.schoenmakers@maastrichtuniversity.nl)** <br>
[Department of Advanced Computing Sciences](https://www.maastrichtuniversity.nl/research/department-advanced-computing-sciences)  
Faculty of Science and Engineering, Maastricht University, The Netherlands

