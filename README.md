# Fuzzy Granule Density-Based Outlier Detection with Multi-Scale Granular Balls
Can Gao, [Xiaofeng Tan](https://xiaofeng-tan.github.io/), Jie Zhou, Weiping Ding, and Witold Pedrycz

This repository is the official implementation of the TKDE submission "**Fuzzy Granule Density-Based Outlier Detection with Multi-Scale Granular Balls**".

<!-- Visit our [**webpage**](https://www.pinlab.org/coskad) for more details. -->
## Abstract
Outlier detection refers to the identification of anomalous samples that deviate significantly from the distribution of normal data and has been extensively studied and used in a variety of practical tasks. However, most unsupervised outlier detection methods are carefully designed to detect specified outliers, while real-world data may be entangled with different types of outliers. In this study, we propose a fuzzy rough sets-based multi-scale outlier detection method to identify various types of outliers. Specifically, a novel fuzzy rough sets-based method that integrates relative fuzzy granule density is first introduced to improve the capability of detecting local outliers. Then, a multi-scale view generation method based on granular-ball computing is proposed to collaboratively identify group outliers at different levels of granularity. Moreover, reliable outliers and inliers determined by the three-way decision are used to train a weighted support vector machine to further improve the performance of outlier detection. The proposed method innovatively transforms unsupervised outlier detection into a semi-supervised classification problem and for the first time explores the fuzzy rough sets-based outlier detection from the perspective of multi-scale granular balls, allowing for high adaptability to different types of outliers. Extensive experiments carried out on both artificial and UCI datasets demonstrate that the proposed outlier detection method significantly outperforms the state-of-the-art methods, improving the results by at least 8.48% in terms of the Area Under the ROC Curve (AUROC) index.

![teaser](assets/1.png)

![teaser](assets/2.png) 

## Content
```
.
├── README.md
├── assets
│   ├── 1.png
│   └── 2.png
├── datasets
│   ├── 15_Hepatitis.npz
│   ├── 28_pendigits.npz
│   ├── 31_satimage-2.npz
│   ├── 35_SpamBase.npz
│   ├── 45_wine.npz
│   ├── 46_WPBC.npz
│   ├── 4_breastw.npz
│   ├── 7_Cardiotocography.npz
│   ├── MVTec-AD_carpet.npz
│   ├── MVTec-AD_metal_nut.npz
│   ├── MVTec-AD_pill.npz
│   ├── arrhythmia.mat
│   ├── autos_variant1.mat
│   ├── cardio.mat
│   ├── chess_nowin_227_variant1.mat
│   ├── ionosphere_b_24_variant1.mat
│   ├── iris_Irisvirginica_11_variant1.mat
│   ├── mammography.mat
│   ├── thyroid_disease_variant1.mat
│   └── wdbc_M_39_variant1.mat
├── main
│   ├── FRS_OD.py
│   ├── GB.py
│   ├── __pycache__
│   │   ├── FRS_OD.cpython-310.pyc
│   │   ├── GB.cpython-310.pyc
│   │   └── units.cpython-310.pyc
│   ├── main.py
│   ├── paramaters.pkl
│   └── units.py
├── paramaters.pkl
├── requirements.txt
└── results
```

## Setup
### Environment
```sh
conda env create -f environment.yml
```

### Datasets
The datasets are selected from [BElloney](https://github.com/BElloney/Outlier-detection) and [ADBench](https://github.com/Minqi824/ADBench), and provided in 
```
./datasets.
```

### **Running** 
To reproduce the results reported in our paper, you can run the code as follows:
``` sh
cd main
python main.py
```
The result will be saved in the folder ./results.

## **Modules**
This project contains the following important modules:
1. ``./main/FRS_OD.py``: the implementation of FRS-based outlier detection methods, and the variant in GB views;
2. ``./main/GB.py``: the implementation of GB generation methods and views update;
3. ``./main/units.py``: some auxiliary function;
4. ``./paramaters.pkl``: the hyperparameter settings.


## Appendix
### The relationship between the number of granular balls and the samples

### The computed complex 


