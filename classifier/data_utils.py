"""
Data utilities for loading and processing Stable Diffusion fault detection data.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.io import savemat

def save_data(data, save_dir='./results', prefix="data", index=None):
    """
    Save data as .npy and .mat formats for MATLAB compatibility.

    Args:
        data: dict, np.ndarray, or list. Data to save, preferably as a dictionary.
        save_dir: str. Directory to save files (default: './results').
        prefix: str. File name prefix (default: 'data').
        index: int. File index, uses timestamp if None.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{prefix}" if index is None else f"{prefix}_{index}"
    npy_path = os.path.join(save_dir, f"{filename}.npy")
    mat_path = os.path.join(save_dir, f"{filename}.mat")

    if isinstance(data, dict):
        data_dict = {k: np.asarray(v) for k, v in data.items()}
    elif isinstance(data, (np.ndarray, list)):
        data_dict = {'data': np.asarray(data)}
    else:
        raise ValueError("Unsupported data type. Use dict, np.ndarray, or list.")

    np.save(npy_path, data_dict, allow_pickle=True)
    savemat(mat_path, data_dict)

def generate_square_sample_coords(n_features, size=96):
    """
    Generate coordinates for sampling a square grid.

    Args:
        n_features: int. Number of features to sample.
        size: int. Size of the square grid (default: 96).

    Returns:
        list. List of (x, y) coordinates.
    """
    step = int(np.sqrt(size * size / n_features))
    coords = [(i, j) for i in range(0, size, step) for j in range(0, size, step)]
    return coords[:n_features]

def analyze_predictions(y_pred, y_test):
    """
    Analyze prediction results, counting totals and matches.

    Args:
        y_pred: np.ndarray. Predicted labels (0 or 1).
        y_test: np.ndarray. True labels (0 or 1).

    Returns:
        dict. Statistics of predictions and true labels.
    """
    if y_pred.shape != y_test.shape:
        raise ValueError("y_pred and y_test must have the same shape")

    return {
        "y_pred_total": y_pred.size,
        "y_pred_0": np.sum(y_pred == 0),
        "y_pred_1": np.sum(y_pred == 1),
        "y_test_total": y_test.size,
        "y_test_0": np.sum(y_test == 0),
        "y_test_1": np.sum(y_test == 1),
        "equal_count": np.sum(y_pred == y_test),
        "pred_0/label_1": np.sum((y_pred == 0) & (y_test == 1)),
        "pred_1/label_0": np.sum((y_pred == 1) & (y_test == 0))
    }

def clean_data(X):
    """
    Clean data by handling NaN, Inf, and large values.

    Args:
        X: pd.DataFrame or np.ndarray. Input feature data.

    Returns:
        np.ndarray. Cleaned data.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    return X.values

def find_errors_with_details(y_pred, y_test, error_types):
    """
    Find prediction errors with detailed information.

    Args:
        y_pred: np.ndarray. Predicted labels.
        y_test: np.ndarray. True labels.
        error_types: list. Error type for each sample.

    Returns:
        list. List of dictionaries with error details.
    """
    if len(y_pred) != len(y_test) or len(y_pred) != len(error_types):
        raise ValueError("y_pred, y_test, and error_types must have the same length.")

    return [
        {"index": i, "error_type": error_types[i], "true_label": true, "predicted_label": pred}
        for i, (pred, true) in enumerate(zip(y_pred, y_test)) if pred != true
    ]

def load_and_process_data(prompt_list, iterations, data_type, n_features, clipscore0, threshold,
                          test_param_types, dataset_dir='./dataset', fixed_latents=True):
    """
    Load and process fault detection data.

    Args:
        prompt_list: list. List of prompt indices.
        iterations: list. List of iteration indices.
        data_type: str. Type of data ('x_t', 'noise_pred', etc.).
        n_features: int. Number of features.
        clipscore0: list. Baseline CLIP scores.
        threshold: float. Threshold for labeling errors.
        test_param_types: list. Parameter types to test.
        dataset_dir: str. Directory containing dataset files.
        fixed_latents: bool. Whether to use fixed latents.

    Returns:
        tuple. (X, y, error_types) - features, labels, and error types.
    """
    xt_filepath = os.path.join(dataset_dir, data_type)
    xt_v1_filepath = os.path.join(dataset_dir, f"{data_type}_v1")
    xt_v2_filepath = os.path.join(dataset_dir, f"{data_type}_v2")
    xt_v3_filepath = os.path.join(dataset_dir, f"{data_type}_v3")
    xt_v4_filepath = os.path.join(dataset_dir, f"{data_type}_v4")

    X, y, error_types = [], [], []
    sample_coords = generate_square_sample_coords(n_features)

    for prompt in prompt_list:
        for i in iterations:
            for blockname in ['down', 'up']:
                for partname in ['blocks_attentions', 'blocks_resnets']:
                    filename = os.path.join(
                        dataset_dir,
                        f"error_clipsore_dict_{blockname}_{partname}.json" if fixed_latents else
                        f"unfixed_error_clipsore_dict_{blockname}_{partname}.json"
                    )
                    if not os.path.exists(filename):
                        continue

                    with open(filename, 'r') as f:
                        error_clipscore_dict = json.load(f)

                    for key, clipscore in error_clipscore_dict.items():
                        error_type = key.split('_', 1)[1]
                        curr_prompt = int(key[1])
                        curr_param = error_type[4:error_type.index('Bit')]
                        if curr_prompt == prompt and curr_param in test_param_types:
                            if curr_prompt in range(5, 10):
                                if 'U' in error_type:
                                    xt_filepath_error = xt_v2_filepath if 'A' in error_type else xt_v4_filepath
                                elif 'D' in error_type:
                                    xt_filepath_error = xt_v1_filepath if 'A' in error_type else xt_v3_filepath
                            else:
                                continue

                            error_path = os.path.join(
                                xt_filepath_error,
                                f"P{prompt}_{data_type}_{error_type}_iter{i}.pth" if fixed_latents else
                                f"P{prompt}_{data_type}_{error_type}_iter{i}.pth"
                            )
                            if not os.path.exists(error_path):
                                continue

                            error_tensor = torch.load(error_path, map_location='cpu')
                            error_data = error_tensor[0, 0, :, :].numpy()  # Adjust slice as needed
                            error_features = error_data.reshape([-1])

                            X.append(error_features)
                            y.append(1 if clipscore < clipscore0[prompt] * threshold else 0)
                            error_types.append(f"{key}:{clipscore}")

    return np.array(X), np.array(y), error_types

def balance_data_by_label_ratio(X, y, error_types, label_ratio):
    """
    Balance dataset to match target label ratio.

    Args:
        X: np.ndarray. Feature data.
        y: np.ndarray. Labels.
        error_types: list. Error types.
        label_ratio: tuple. (label_0_ratio, label_1_ratio).

    Returns:
        tuple. Balanced (X, y, error_types).
    """
    label_0_ratio, label_1_ratio = label_ratio
    X_0 = X[y == 0]
    y_0 = y[y == 0]
    error_types_0 = [error_types[i] for i in range(len(y)) if y[i] == 0]
    X_1 = X[y == 1]
    y_1 = y[y == 1]
    error_types_1 = [error_types[i] for i in range(len(y)) if y[i] == 1]

    total_samples = len(y)
    target_0_count = int(label_0_ratio * total_samples)
    target_1_count = total_samples - target_0_count

    X_0_balanced = X_0[:target_0_count]
    y_0_balanced = y_0[:target_0_count]
    error_types_0_balanced = error_types_0[:target_0_count]
    X_1_balanced = X_1[:target_1_count]
    y_1_balanced = y_1[:target_1_count]
    error_types_1_balanced = error_types_1[:target_1_count]

    X_balanced = np.concatenate((X_0_balanced, X_1_balanced), axis=0)
    y_balanced = np.concatenate((y_0_balanced, y_1_balanced), axis=0)
    error_types_balanced = error_types_0_balanced + error_types_1_balanced

    indices = np.random.permutation(len(y_balanced))
    return X_balanced[indices], y_balanced[indices], [error_types_balanced[i] for i in indices]