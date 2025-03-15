"""
This script evaluates the performance of multiple anomaly detection models across different datasets,
including both stable and unstable models. Various evaluation metrics are used to compare model performance,
and the results are stored in Excel files.

### Main Features:
- Load datasets (supports `.npz` and `.mat` formats)
- Train multiple anomaly detection models
- Compute and store evaluation metrics
- Record the results of stable and unstable models
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from mat4py import loadmat
from pyod.utils.utility import get_label_n

# Importing anomaly detection models from the pyod library
# The models selected for testing include a mix of stable and unstable methods
# to assess their performance across different datasets.
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.kde import KDE
from pyod.models.loda import LODA


def load_data(path):
    """
    Load dataset from a given file path, supporting `.npz` and `.mat` formats.

    Parameters:
    - path (str): The file path of the dataset.

    Returns:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Label vector.
    """
    try:
        data = loadmat(path)
    except:
        data = np.load(path, allow_pickle=True)
    try:
        data = np.array(data['trandata'])
    except:
        X = np.array(data['X'])
        y = np.array(data['y'])
        return X, y

    X = data[:, 0:-1]
    y = data[:, -1]
    if max(y) != 1:
        y -= min(y)
    if sum(y) > len(y) / 2:
        y = np.where(y == 0, 1, 0)
    return X, y


# Define the storage paths for result files
auc_excel_file = 'AUC_Results.xlsx'
pre_excel_file = 'Precision_Results.xlsx'
rec_excel_file = 'Recall_Results.xlsx'

# Dataset folder path
# Put the test datasets in a folder
dataset_folder = 'datasets'

# Define the model collections
stable_models = {
    'HBOS': HBOS,  # 0
    'KDE': KDE,  # 1
    'LOF': LOF,  # 2
    'PCA': PCA,  # 3
}

unstable_models = {
    'OCSVM': OCSVM,  # 4
    'KNN': KNN,  # 5
    'Feature Bagging': FeatureBagging,  # 6
    'LODA': LODA,  # 7
    'IForest': IForest,  # 8
    'INNE': INNE,  # 9
}

# Initialize result storage
auc_results = []
pre_results = []
rec_results = []
auc_df = pd.DataFrame(
    columns=['Dataset', 'HBOS', 'KDE', 'LOF', 'PCA', 'OCSVM', 'KNN', 'Feature Bagging', 'LODA', 'IForest', 'INNE'])
pre_df = pd.DataFrame(
    columns=['Dataset', 'HBOS', 'KDE', 'LOF', 'PCA', 'OCSVM', 'KNN', 'Feature Bagging', 'LODA', 'IForest', 'INNE'])
rec_df = pd.DataFrame(
    columns=['Dataset', 'HBOS', 'KDE', 'LOF', 'PCA', 'OCSVM', 'KNN', 'Feature Bagging', 'LODA', 'IForest', 'INNE'])

# Retrieve all dataset files from the folder
dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith('.npz') or f.endswith('.mat')]

# Set the contamination rate (proportion of outliers)
contamination = 0.05

# Iterate over each dataset file
for dataset_name in dataset_files:
    auc, pre, rec = [], [], []
    file_path = os.path.join(dataset_folder, dataset_name)
    X, y = load_data(file_path)
    # Normalize the dataset (Optional)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Set the number of outliers
    # The number of outliers is calculated based on a predefined contamination
    num_outliers = len(y) * contamination
    print(f"Processing dataset: {dataset_name}")

    # Test stable models
    for model_name, model_class in stable_models.items():
        model = model_class(contamination=contamination)
        model.fit(X)
        try:
            predict_score = model.decision_function(X)
            y_pred = get_label_n(y, predict_score, num_outliers)
            auc_ = roc_auc_score(y, predict_score)
            pre_ = precision_score(y, y_pred)
            rec_ = recall_score(y, y_pred)
        except ValueError as e:
            print(f"Error for model {model_name} on dataset {dataset_name}: {str(e)}")
            continue

        # Store results
        auc.append(auc_)
        pre.append(pre_)
        rec.append(rec_)

        print(f"{model_name} - Dataset: {dataset_name} | AUC: {auc_}, Precision: {pre_}, Recall: {rec_}")

    # Test unstable models 10 times and compute the average results
    for model_name, model_class in unstable_models.items():
        model = model_class(contamination=contamination)
        auc_scores, pre_scores, rec_scores = [], [], []

        for _ in range(10):
            model.fit(X)
            try:
                predict_score = model.decision_function(X)
                y_pred = get_label_n(y, predict_score, num_outliers)
                auc_scores.append(roc_auc_score(y, predict_score))
                pre_scores.append(precision_score(y, y_pred))
                rec_scores.append(recall_score(y, y_pred))
            except ValueError as e:
                print(f"Error for model {model_name} on dataset {dataset_name}: {str(e)}")
                continue

        avg_auc = np.mean(auc_scores) if auc_scores else None
        avg_pre = np.mean(pre_scores) if pre_scores else None
        avg_rec = np.mean(rec_scores) if rec_scores else None

        # Store results
        auc.append(avg_auc)
        pre.append(avg_pre)
        rec.append(avg_rec)

        print(
            f"{model_name} - Dataset: {dataset_name} | Average AUC: {avg_auc}, Average Precision: {avg_pre}, Average Recall: {avg_rec}")

    # Create DataFrames
    new_auc_df = pd.DataFrame(
        {'Dataset': dataset_name, 'HBOS': auc[0], 'KDE': auc[1], 'LOF': auc[2], 'PCA': auc[3], 'OCSVM': auc[4],
         'KNN': auc[5], 'Feature Bagging': auc[6], 'LODA': auc[7], 'IForest': auc[8], 'INNE': auc[9]},
        index=['Dataset'])
    new_pre_df = pd.DataFrame(
        {'Dataset': dataset_name, 'HBOS': pre[0], 'KDE': pre[1], 'LOF': pre[2], 'PCA': pre[3], 'OCSVM': pre[4],
         'KNN': pre[5], 'Feature Bagging': pre[6], 'LODA': pre[7], 'IForest': pre[8], 'INNE': pre[9]},
        index=['Dataset'])
    new_rec_df = pd.DataFrame(
        {'Dataset': dataset_name, 'HBOS': rec[0], 'KDE': rec[1], 'LOF': rec[2], 'PCA': rec[3], 'OCSVM': rec[4],
         'KNN': rec[5], 'Feature Bagging': rec[6], 'LODA': rec[7], 'IForest': rec[8], 'INNE': rec[9]}, index=['Dataset'])

    auc_df = pd.concat([auc_df, new_auc_df], axis=0)
    pre_df = pd.concat([pre_df, new_pre_df], axis=0)
    rec_df = pd.concat([rec_df, new_rec_df], axis=0)

    # Save results directly to Excel files
    auc_df.to_excel(auc_excel_file, sheet_name='AUC Results', index=False)
    pre_df.to_excel(pre_excel_file, sheet_name='Precision Results', index=False)
    rec_df.to_excel(rec_excel_file, sheet_name='Recall Results', index=False)
