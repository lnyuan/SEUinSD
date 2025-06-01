"""
Main script for running fault detection classification experiments.
"""

import os
import sys
import time
from .data_utils import load_and_process_data
from .model_utils import train_adaboost
from .visualization_utils import plot_data_distribution, plot_data_pdf, plot_distribution_with_labels

def run_experiment(
    type_name='xt',
    iterations=None, [1],
    n_slice=0,
    threshold=0.96,
    n_estimators=[39,40, 40, 41],
    learning_rates=[10],
    test_param_types=['attn1_v', 'fc1', 'attn2_v'],
    dataset_root='./dataset',
    result_dir='error_detect_results',
    data_mode='only_fixed',
    train_label0_ratio=0.4,
    test_label0_ratio=0.4,
    label_ratio=None,
    show_fig=True,
    data_distribution=False
):
    """
    Run fault detection classification experiment.

    Args:
        type_name: str. Data type name ('xt', 'ut', etc.).
        iterations: list. Iteration indices.
        n_slice: int. Data slice index.
        threshold: float. CLIP score threshold.
        n_estimators: list. Number of AdaBoost estimators.
        learning_rates: list. Learning rates.
        test_param_types: list. list. Parameter types to test.
        dataset_root: str. Root directory of dataset.
        result_dir: str. Directory to save results.
        data_mode: str. Data mode ('all', 'only_fixed', 'only_unfixed').
        label_ratio: float or None. Overall label ratio if not None.
        train_label0_ratio: float. Label ratio for training set.
        test_label0_ratio: float. Label ratio for test set.
        show_fig: bool. Whether to show figures.
        data_distribution: bool. Whether to plot data distributions.

    Returns:
        None
    """
    # Data type mapping
    type_name_dict = {
        'xt': 'x_t',
        'ut': 'noise_pred',
        'ut_text': 'noise_pred_text',
        'ut_null': 'noise_pred_uncond'
    }

    # CLIP baseline scores
    clipscore0 = [33.96015, 26.03592, 32.313862, 28.252852, 30.555044, 27

    # Create result directory
    result_dir = os.path.join(result_dir, f'{type_name}_results')
    os.makedirs(result_dir, exist_ok=True)

    # Redirect stdout to log file
    class TimestampedLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.logfile = open(filename, "a")

        def write(self, message):
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            self.terminal.write(f"{message}\n")
            self.logfile.write(f"{timestamp} {message}\n")

        def flush(self):
            self.terminal.flush()
            self.logfile.flush()

    sys.stdout = TimestampedLogger(f"{result_dir}/{os.path.basename(__file__)[:-3]}_log.txt")

    # Load and process data
    prompt_list = list(range(5))
    n_features = 96 * 96
    X, y, error_types = load_and_process_data(
        prompt_list, iterations, type_name_dict[type_name], n_features, clipscore0, threshold,
        test_param_types, dataset_dir=dataset_root, fixed_latents=True
    )

    # Balance overall label ratio
    if label_ratio:
        from .data_utils import balance_data_by_label_ratio
        X, y, error_types = balance_data_by_label_ratio(X, y, error_types, [label_ratio, 1-label_ratio])

    # Plot data distributions
    if data_distribution:
        plot_distribution_with_labels(X, y, n_slice, result_dir, type_name)
        plot_data_distribution(X, y, type_name, n_slice, result_dir, 'hold on')
        plot_data_pdf(X, y, type_name, n_slice, result_dir, 'hold on')

    # Train and evaluate classifier
    train_adaboost(
        X, y, error_types, n_estimators, learning_rates, test_size=0.2,
        train_label0_ratio, test_label0_ratio, result_dir, type_name, n_slice,
        data_mode, show_fig
    )

if __name__ == "__main__":
    run_experiment(
        type_name='xt',
        iterations=[1],
        n_slice=0,
        threshold=0.96,
        n_estimators=range(39, 41),
        learning_rates=[10],
        test_param_types=['attn1_v', 'fc1', 'attn2_v'],
        data_distribution=True
    )