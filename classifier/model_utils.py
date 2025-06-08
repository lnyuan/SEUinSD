"""
Model utilities for training and evaluating fault detection classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, zero_one_loss, classification_report
from sklearn.model_selection import train_test_split
from .data_utils import clean_data, analyze_predictions, find_errors_with_details, save_data
import joblib
import datetime
import os

def train_adaboost(X, y, error_types, test_n_estimators, learning_rates, test_size, train_label0_ratio,
                   test_label0_ratio, result_dir, type_name, n_slice, curr_data_mode, show_fig=False):
    """
    Train and evaluate an AdaBoost classifier for fault detection.

    Args:
        X: np.ndarray. Feature data.
        y: np.ndarray. Labels.
        error_types: list. Error types for each sample.
        test_n_estimators: list. Range of n_estimators to test.
        learning_rates: list. Learning rates to test.
        test_size: float. Proportion of test set.
        train_label0_ratio: float. Target ratio of label 0 in training set.
        test_label0_ratio: float. Target ratio of label 0 in test set.
        result_dir: str. Directory to save results.
        type_name: str. Name of data type (e.g., 'xt').
        n_slice: int. Data slice index.
        curr_data_mode: str. Data mode ('only_fixed', 'only_unfixed', 'all').
        show_fig: bool. Whether to display figures.

    Returns:
        None
    """
    current_run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(result_dir, exist_ok=True)

    # Split dataset
    X_train, X_test, y_train, y_test, error_types_train, error_types_test = train_test_split(
        X, y, error_types, test_size=test_size, random_state=42, stratify=y
    )

    # Balance label ratios
    if train_label0_ratio:
        from .data_utils import balance_data_by_label_ratio
        X_train, y_train, error_types_train = balance_data_by_label_ratio(
            X_train, y_train, error_types_train, [train_label0_ratio, 1 - train_label0_ratio]
        )
    if test_label0_ratio:
        from .data_utils import balance_data_by_label_ratio
        X_test, y_test, error_types_test = balance_data_by_label_ratio(
            X_test, y_test, error_types_test, [test_label0_ratio, 1 - test_label0_ratio]
        )

    # Clean data
    X_train = clean_data(X_train)
    X_test = clean_data(X_test)

    # Train and evaluate
    best_accuracy = 0
    best_estimator = None
    best_n = 0
    best_learning_rate = 0
    train_losses = []
    test_losses = []
    all_accuracy = []

    for learning_rate in learning_rates:
        train_loss = []
        test_loss = []
        acc = []
        for n_estimators in test_n_estimators:
            clf = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate / 10,
                algorithm="SAMME"
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            acc.append(accuracy)
            print(f"Lr={learning_rate/10}, Ne={n_estimators}, acc={accuracy}, {analyze_predictions(y_pred, y_test)}")
            train_loss.append(zero_one_loss(y_train, clf.predict(X_train)))
            test_loss.append(zero_one_loss(y_test, y_pred))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pred = analyze_predictions(y_pred, y_test)
                best_classification_report = classification_report(y_test, y_pred)
                best_estimator = clf
                best_n = n_estimators
                best_learning_rate = learning_rate / 10
                best_train_losses = [zero_one_loss(y_train, y_pred) for y_pred in clf.staged_predict(X_train)]
                best_test_losses = [zero_one_loss(y_test, y_pred) for y_pred in clf.staged_predict(X_test)]
                best_pred_inacc = find_errors_with_details(y_pred, y_test, error_types_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        all_accuracy.append(acc)

    # Save results
    dataset_label_info = f"Train0({np.sum(y_train==0)/y_train.shape[0]:.4f}){np.sum(y_train==0)}of{y_train.shape[0]}" \
                        f"Test0({np.sum(y_test==0)/y_test.shape[0]:.4f}){np.sum(y_test==0)}of{y_test.shape[0]}"
    data_info = f"{curr_data_mode}{type_name}slice{n_slice}"
    test_param_info = f"Ne{test_n_estimators}Lr{learning_rates}"
    result_name_main = f"adaboost_[{test_param_info}_{dataset_label_info}]_[{data_info}]"

    joblib.dump(best_estimator, f'{result_dir}/{result_name_main}_bestclf_ACC{best_accuracy:.4f}_{current_run_time}.pkl')
    save_data(train_losses, save_dir=result_dir, prefix=f'{result_name_main}_train_losses_{current_run_time}')
    save_data(test_losses, save_dir=result_dir, prefix=f'{result_name_main}_test_losses_{current_run_time}')
    save_data(all_accuracy, save_dir=result_dir, prefix=f'{result_name_main}_accuracy_{current_run_time}')

    # Plot losses
    plt.figure(figsize=(len(test_n_estimators), 8))
    colors = plt.cm.tab10(range(len(learning_rates)))
    for lr_idx, lr in enumerate(learning_rates):
        plt.plot(test_n_estimators, train_losses[lr_idx], label=f'Train Loss, Lr={lr/10}', marker='o', color=colors[lr_idx])
        plt.plot(test_n_estimators, test_losses[lr_idx], label=f'Test Loss, Lr={lr/10}', marker='^', color=colors[lr_idx])
    plt.xticks(test_n_estimators)
    plt.xlabel('Number of Weak Classifiers')
    plt.ylabel('Zero-One Loss')
    plt.title(f'AdaBoost Loss: {result_name_main}')
    plt.legend()
    plt.savefig(f"{result_dir}/{result_name_main}_Loss_{current_run_time}.svg")
    if show_fig:
        plt.show()
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(len(test_n_estimators), 8))
    for lr_idx, lr in enumerate(learning_rates):
        plt.plot(test_n_estimators, all_accuracy[lr_idx], label=f'Lr={lr/10}', marker='o', color=colors[lr_idx])
        for j, acc in enumerate(all_accuracy[lr_idx]):
            plt.text(test_n_estimators[j], acc, f"{acc:.4f}", ha='center', va='bottom')
    plt.xticks(test_n_estimators)
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.title(f'AdaBoost Accuracy: {result_name_main}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{result_dir}/{result_name_main}_Acc_{current_run_time}.svg")
    if show_fig:
        plt.show()
    plt.close()

    print(f"Best Accuracy: {best_accuracy:.4f}, n_estimators: {best_n}, learning_rate: {best_learning_rate}")
    print(f"Best Prediction: {best_pred}")
    print(f"Classification Report:\n{best_classification_report}")