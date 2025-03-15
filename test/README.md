
# Unsupervised Outlier Detection Model Evaluation

## Project Overview

This script is used to evaluate the performance of multiple unsupervised outlier detection models across different datasets. It supports the evaluation of both stable and unstable models, using multiple evaluation metrics such as AUC (Area Under the Curve), Precision, and Recall. The results are stored in Excel files for easy analysis and comparison.

## Features

- **Dataset Support**: Supports loading datasets in `.npz` and `.mat` formats sourced from publicly available repositories.
- **Multi-Model Evaluation**: Evaluates the performance of various unsupervised outlier detection models.
- **Multiple Metrics**: Computes evaluation metrics such as AUC, Precision, and Recall.
- **Result Storage**: Saves performance results for each model on different datasets in Excel files.

## Requirements

Before running the script, make sure the following Python libraries are installed:

- `numpy`
- `pandas`
- `pyod`
- `scikit-learn`
- `mat4py`

You can set up the environment by installing the given requirements:

```bash
pip install -r requirements.txt
```

## File Structure

Ensure that your project directory structure looks like this:

```
project_directory/
│
├── datasets/                  # Folder containing datasets (e.g., .npz or .mat files)
│   ├── dataset1.mat
│   ├── dataset2.npz
│   └── ...
├── unsupervised_test.py  # Script file
└── AUC_Results.xlsx            # AUC result file
└── Precision_Results.xlsx      # Precision result file
└── Recall_Results.xlsx        # Recall result file
```

## Run the Script

Run the script by executing the following command in the terminal or command prompt:

```bash
python unsupervised_test.py
```

The script will automatically process all datasets in the `datasets/` folder, evaluate multiple unsupervised outlier detection models, and store the results in Excel files.

## Customizing the Script

The script can be customized, for example:

- **Add Datasets**: Add new `.npz` or `.mat` files to the `datasets/` folder.
- **Modify Contamination Rate**: Adjust the `contamination` value in the script to control the proportion of outliers in the dataset.
- **Add New Models**: You can import and add additional unsupervised outlier detection models (e.g., from `pyod` library).

