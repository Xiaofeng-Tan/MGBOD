<h1 align="center"><strong>Fuzzy Granule Density-Based Outlier Detection with Multi-Scale Granular Balls</strong></h1>
<p align="center">
  Can Gao<sup>1</sup>&emsp;
  <a href='https://xiaofeng-tan.github.io/' target='_blank'>Xiaofeng Tan<sup>1</sup></a>&emsp;
  Jie Zhou<sup>1</sup>&emsp;
  Weiping Ding<sup>2</sup>&emsp;
  Witold Pedrycz<sup>3</sup>&emsp;
  <br>
  <sup>1</sup>Shenzhen University&emsp;
  <sup>2</sup>Nantong University&emsp;
  <sup>3</sup>University of Alberta&emsp;
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/abstract/document/10821488">
    <img src="https://img.shields.io/badge/TKDE-2025-0066CC" alt="TKDE 2025">
  </a>
  <a href="https://ieeexplore.ieee.org/abstract/document/10821488">
    <img src="https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow" alt="Paper PDF">
  </a>
  <a href="https://github.com/Xiaofeng-Tan/MGBOD">
    <img src="https://img.shields.io/badge/Code-GitHub-green?style=flat&logo=GitHub&logoColor=green" alt="GitHub Repository">
  </a>
</p>

This repository provides the official implementation of the TKDE 2025 paper "**Fuzzy Granule Density-Based Outlier Detection with Multi-Scale Granular Balls**". For any questions, please feel free to contact us via 📧 email (xiaofengtan@seu.edu.cn) or WeChat (txf_06_20)!

## 🎉 News
- **2025-12-16**: **Major refactoring** - Reorganized project with cleaner structure: `src/` for core code, `scripts/` for executables, `configs/` for parameters, `docs/` for documentation. Removed legacy `main/` directory.
- **2025-11-12**: Released **visualization codes** in [`visualize.py`](./visualize.py). Thanks to Peng Dai for his valuable suggestion.
- **2025-03-15**: Released codes for **reproducing other baselines** in the [`test/`](./test) folder. Thanks to [@Zhiyu Chen](https://github.com/czy629).
- **2024-12-24**: Our paper has been accepted by IEEE Transactions on Knowledge and Data Engineering.
- **2024-09-13**: Released the main experiment codes.

## 🌟 Related Works
Here are some notable outlier detection works based on Granular Balls:

- **A Kernelized Fuzzy Approximation Fusion Model with Granular-ball Computing for Outlier Detection** (Information Fusion) [[pdf](https://www.sciencedirect.com/science/article/pii/S1566253525007729)] [[code](https://github.com/LYXRhythm/KFGOD)]
- **Identifying Outliers via Local Granular-Ball Density** (TNNLS) [[pdf](https://ieeexplore.ieee.org/abstract/document/11073180)] [[code](https://github.com/Mxeron/GBDO)]
- **GBMOD: A granular-ball mean-shift outlier detector** (PR) [[pdf](https://www.sciencedirect.com/science/article/pii/S0031320324008665)] [[code](https://github.com/cstzsthl/GBMOD)]
- **Granular-ball computing-based Random Walk for anomaly detection** (PR) [[pdf](https://www.sciencedirect.com/science/article/pii/S0031320325002481)] [[code](https://github.com/optimusprimeyy/GBRAD)]
- **GBNOD: Granular-ball neighborhood outlier detection** (Neurocomputing) [[pdf](https://www.sciencedirect.com/science/article/abs/pii/S0925231225031017)] [[code](https://github.com/Mxeron/GBNOD)]

If any relevant work is missing, please feel free to open an issue or contact us via email (txf0620@gmail.com) or WeChat (txf_06_20). We greatly appreciate the contributions from the research community. 🌹

## Abstract
Outlier detection involves identifying anomalous samples that significantly deviate from the distribution of normal data, a task that has been extensively studied and applied in various practical scenarios. However, most unsupervised outlier detection methods are specifically designed to detect particular types of outliers, whereas real-world data often contains multiple outlier types simultaneously. In this study, we propose a fuzzy rough sets-based multi-scale outlier detection method capable of identifying diverse outlier types. Specifically, we first introduce a novel fuzzy rough sets approach that incorporates relative fuzzy granule density to enhance the detection of local outliers. We then propose a multi-scale view generation method based on granular-ball computing to collaboratively identify group outliers at different granularity levels. Furthermore, we utilize reliable outliers and inliers determined by three-way decision to train a weighted support vector machine, thereby improving outlier detection performance. Our method innovatively transforms unsupervised outlier detection into a semi-supervised classification problem and represents the first exploration of fuzzy rough sets-based outlier detection from a multi-scale granular balls perspective, offering high adaptability to various outlier types. Extensive experiments on both synthetic and UCI datasets demonstrate that our proposed method significantly outperforms state-of-the-art approaches, achieving at least 8.48% improvement in terms of the Area Under the ROC Curve (AUROC) metric.

![teaser](assets/1.png)
![teaser](assets/2.png)

## Project Structure
```
MGBOD/
├── run.py                    # Quick start: run experiments
├── visualize.py              # Quick start: run visualization
├── requirements.txt          # Python dependencies
├── README.md
│
├── configs/                  # Configuration files
│   └── parameters.pkl        # Hyperparameter settings (δ, λ)
│
├── scripts/                  # Executable scripts
│   ├── run_experiment.py     # Main experiment script
│   └── run_visualization.py  # Granular ball visualization
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── detector.py           # High-level MGBOD API
│   ├── core/                 # Core algorithms
│   │   ├── frs_od.py         # FRS-based outlier detection
│   │   └── granular_ball.py  # Granular ball generation
│   ├── utils/                # Utilities
│   │   ├── data.py           # Data loading
│   │   └── metrics.py        # Evaluation metrics
│   └── visualization/        # Plotting
│       └── plot.py           # GB visualization
│
├── datasets/                 # Dataset files (.npz, .mat)
├── results/                  # Experiment outputs
├── figures/                  # Visualization outputs
├── docs/                     # Documentation & appendix
│   ├── Appendix_A_Relationship_Analysis.pdf
│   └── The detailed description of datasets.pdf
├── assets/                   # README images
└── test/                     # Baseline comparison scripts
```

## Setup
### Environment

**Python Version**: Python 3.8+ recommended

**Option 1: Using pip (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: Using conda**
```bash
conda create -n mgbod python=3.10
conda activate mgbod
pip install -r requirements.txt
```

**Required Dependencies:**
| Package | Version | Description |
|---------|---------|-------------|
| numpy | >=1.21.0 | Numerical computing |
| scipy | >=1.7.0 | Scientific computing |
| torch | >=1.10.0 | Tensor operations |
| scikit-learn | >=1.0.0 | Machine learning (SVM) |
| pandas | >=1.3.0 | Data manipulation |
| openpyxl | >=3.0.0 | Excel file export |
| mat4py | >=0.5.0 | MATLAB file loading |
| matplotlib | >=3.4.0 | Visualization (optional) |

### Datasets
The datasets are sourced from [BElloney](https://github.com/BElloney/Outlier-detection) and [ADBench](https://github.com/Minqi824/ADBench), and are provided in the `./datasets` directory.

| Dataset | Description |
|---------|-------------|
| Arrhythmia | Samples from minority classes 3, 4, 5, 7, 8, 9, 14, and 15 are grouped as outliers, while remaining samples are considered inliers. |
| Autos | Samples from "-2" and "-1" classes are treated as outliers, with other classes considered as inliers. |
| Breast | Samples in the "malignant" class are outliers, while "benign" class samples are inliers. |
| Cardio | The "pathologic" class is downsampled to 176 samples as outliers. "Normal" class samples are inliers, and "suspect" class samples are removed. |
| Cardiotocography | Classes "2" and "3" are downsampled to 33 samples as outliers, with other classes as inliers. |
| Carpet | A category from MVTec AD datasets containing both normal and defective carpet samples. |
| Chess | The "nowin" class is downsampled to 227 samples as outliers, with remaining class samples as inliers. |
| Hepatitis | "Hepatitis" class samples are outliers, while "non-hepatitis" class samples are inliers. |
| Ionosphere | Class "b" is downsampled to 24 anomalous samples, with remaining class samples as inliers. |
| Iris | "Iris-virginica" class is downsampled to 11 anomalous samples, with other classes as inliers. |
| Mammography | "Calcification" class samples are outliers, while other samples are inliers. |
| Metal Nut | A category from MVTec AD datasets containing both normal and defective metal samples. |
| Pendigits | "0" class samples are outliers, with remaining digit samples (1-9) as inliers. |
| Pill | A category from MVTec AD datasets containing both normal and defective pill samples. |
| Satimage | "2" class is downsampled to 71 anomalous samples, with other classes combined as inliers. |
| Spam | "Spam" class samples are outliers, while "non-spam" class samples are inliers. |
| Thyroid | "Hyperfunction" class samples are outliers, with normal and subnormal function samples as inliers. |
| WDBC | "M" class is downsampled to 39 samples as outliers, with other samples as inliers. |
| Wine | "1" class is downsampled to 10 samples as outliers, with "2" and "3" class samples as inliers. |
| WPBC | "R" (minority) class samples are outliers, with other samples as inliers. |

## Running
To reproduce the results reported in our paper, run the following commands:

```bash
# Quick start
python run.py

# Or run the script directly
python scripts/run_experiment.py
```

The results will be saved in the `./results` directory.

## Visualization

```bash
# Quick start
python visualize.py

# Or run the script directly
python scripts/run_visualization.py
```

The figures will be saved in the `./figures` directory.

## Modules
This project includes the following key modules:

| Module | Description |
|--------|-------------|
| `src/core/frs_od.py` | FRS-based outlier detection (FRS_OD, FRS_OD_GB classes) |
| `src/core/granular_ball.py` | Granular ball generation and multi-scale view updates |
| `src/detector.py` | High-level API: `fit()`, `run_FRS()`, `OD_GB()` |
| `src/utils/data.py` | Dataset loading for .npz and .mat formats |
| `src/utils/metrics.py` | Evaluation metrics and group scoring |
| `src/visualization/plot.py` | Granular ball visualization |
| `configs/parameters.pkl` | Pre-tuned hyperparameters (δ, λ) for each dataset |

## Appendix
Please refer to the following files for additional details:
- [The detailed description of datasets.pdf](./docs/The%20detailed%20description%20of%20datasets.pdf)
- [Appendix A: Relationship Analysis.pdf](./docs/Appendix_A_Relationship_Analysis.pdf)

## Acknowledgement
This work builds upon several excellent research works and open-source projects. We sincerely thank all the authors for their contributions:

- https://github.com/Minqi824/ADBench

## Citation
If you find this repository helpful in your research, please consider citing our paper and starring the repository ⭐.

```bibtex
@ARTICLE{10821488,
  author={Gao, Can and Tan, Xiaofeng and Zhou, Jie and Ding, Weiping and Pedrycz, Witold},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Fuzzy Granule Density-Based Outlier Detection with Multi-Scale Granular Balls}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Outlier detection;fuzzy rough sets;fuzzy granule density;multi-scale granular balls;three-way decision},
  doi={10.1109/TKDE.2024.3525003}}
```
