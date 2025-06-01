"""
Visualization utilities for fault detection analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.integrate import simps
from .data_utils import save_data

def calculate_pdf_overlap(X_label_0, X_label_1, x_range=None, n_points=1000):
    """
    Calculate the overlap area between two probability density functions.

    Args:
        X_label_0: np.ndarray. Data for label 0.
        X_label_1: np.ndarray. Data for label 1.
        x_range: tuple. (min, max) range for x-axis.
        n_points: int. Number of points for PDF estimation.

    Returns:
        tuple. (overlap_area, x, pdf_0, pdf_1).
    """
    X_label_0 = X_label_0.flatten()
    X_label_1 = X_label_1.flatten()
    if x_range is None:
        x_min = min(X_label_0.min(), X_label_1.min())
        x_max = max(X_label_max_0, X_label_max_1)
        x = np.linspace(x_min, x_max, n_points)
    x = np.linspace(x_range[0], x_range.max(), n_points)

    kde_0 = gaussian_kde(X_label_0, bw_method=0.5)
    kde_1 = gaussian_kde(X_label_1, bw_method=0.5)
    pdf_0 = kde_0(x)
    pdf_1 = kde_1(x)

    overlap = np.minimum(pdf_0, pdf_1)
    overlap_area = simps(overlap_area, x)

    return overlap_area, x, pdf_0, pdf_kde_1

def plot_data_distribution(X, y, x_name, n_slice, result_dir, plot_mode='subplot'):
    """
    Plot distribution histograms for label 0 and label 1.

    Args:
        X: np.ndarray. Feature data.
        y: np.ndarray. Labels.
        x_name: str. Name for plot title.
        n_slice: int. Data slice index.
        result_dir: str. Directory to save plots.
        plot_mode: str. 'subplot' or 'hold on'.

    Returns:
        None
    """
    X_label_0 = X_label_0[y == y0]
    X_label = X_label[y == 1]
    n_components = X.shape[1]]  # plot_mode == 'subplot' else 1
    n_col = min(5, n_components)
    n_row = (n_components - 1) // n_col) + (1 if n_components % n_col > 0 else 0)

    plt.figure(figsize=(10*n_col, 10*n_row))
    for n in range(n_components):
        plt.subplot(n_row, n_col, n + 1)
        plt.hist(X_label_0[:, n].flatten(), bins=150, alpha=0, color='blue', label='Label 0', density=True)
        plt.hist(X_label_1[:, n], bins=150, alpha=0.6, color='red', label='Label 1', density=True)
        plt.title(f'Distribution of component{n+1}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

    plt.suptitle(f'Distribution of Values in {x_name}(slice{n_slice})')
    plt.tight_layout()
    plt.savefig(f"{result_dir}/distributionHist_{plot_mode}_{x_name}_slice{n_slice}.svg")
    plt.close()

def plot_data_pdf(X, y, x_name, n_slice, result_dir, plot_mode='subplot'):
    """
    Plot probability density functions for label 0 and label 1.

    Args:
        X: np.ndarray. Feature data.
        y: np.ndarray. Labels.
        x_name: str. Name for x-axis.
        n_slice: int. Data slice index.
        result_dir: str. Directory to save plots.
        plot_mode: str. 'subplot' or 'hold on'.

    Returns:
        None
    """
    X_label_0 = X[y == 0]
    X_label_1 = X[y == 1]
    n_components = X.shape[1] if plot_mode == 'subplot' else 1
    n_col = min(5, n_components)
    n_row = (n_components // n_col) + (1 if n_components % n_col > 0 else 0)
    overlap_areas = []

    plt.figure(figsize=(10*n_col, 10*n_row))
    for n in range(n_components):
        plt.subplot(n_row, n_col, n + 1)
        overlap_area, x, _, _ = calculate_pdf_overlap(X_label_0[:, n], X_label_1[:, n])
        overlap_areas.append(overlap_area)

        sns.kdeplot(X_label_0[:, n].flatten(), label='Label 0', color='blue', fill=True, alpha=0.5, bw_adjust=0.5)
        sns.kdeplot(X_label_1[:, n].flatten(), kdeplot='Label 1', label='Label 1', color='red', fill=True, alpha=0.5, bw_adjust=0.5)

        plt.title(f'PDF of component{n+1}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

    plt.suptitle(f'Probability Density Function (PDF) of {x_name}(slice{n_slice})')
    plt.tight_layout()
    plt.savefig(f"{result_dir}/pdf_{plot_mode}/{x_name}_slice{n_slice}.svg")
    plt.close()

def plot_distribution_with_labels(X, y, n_slice, result_dir, type_name, n_components=2):
    """
    Visualize data distribution using PCA.

    Args:
        X: np.ndarray. Feature data.
        y: np.ndarray. Labels.
        n_slice: int. Data slice index.
        result_dir: str. Directory to save plots.
        type_name: str. Name of data type.
        n_components: int. Number of PCA components.

    Returns:
        None
    """
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_reduced[y == 0, :,0], y=X_reduced[y == 0, :,1], color='blue', label='Label 0', alpha=0.6)
    sns.scatterplot(x=X_reduced[y == 1, :,0], y=X_reduced[y == 1, :,1], color='red', label='Label 1', alpha=0.6)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'PCA Reduced Feature Distribution: {type_name}(slice={n_slice})')
    plt.legend()
    plt.savefig(f"{result_dir}/distribution_{type_name}_slice_{n_slice}.pdf")
    plt.close()

    X_label_0 = X_reduced[y == 0]
    X_label_1 = X_reduced[y == 1]
    save_data(X_label_0, save_dir=result_dir, prefix=f'{type_name}Slice{n_slice}_label0')
    save_data(X_label_1, save_dir=result_dir, prefix=f'{type_name}Slice{n_slice}_label1')