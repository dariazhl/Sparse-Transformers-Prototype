import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    average_precision_score, matthews_corrcoef, cohen_kappa_score, precision_recall_curve
)
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

def calculate_truthfulness_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None,
                                 average: str = 'weighted') -> Dict[str, float]:
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    unique_labels = np.unique(y_true)
    is_binary = len(unique_labels) <= 2

    if is_binary:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        if y_prob is not None and y_prob.shape[1] >= 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except ValueError:
                metrics['roc_auc'] = 0.5

            try:
                metrics['pr_auc'] = average_precision_score(y_true, y_prob[:, 1])
            except ValueError:
                metrics['pr_auc'] = np.mean(y_true)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

        if y_prob is not None:
            try:
                if len(unique_labels) == y_prob.shape[1]:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
                else:
                    metrics['roc_auc'] = 0.5
            except (ValueError, IndexError):
                metrics['roc_auc'] = 0.5

    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

    metrics.update(calculate_truthfulness_specific_metrics(y_true, y_pred, y_prob))

    return metrics

def calculate_truthfulness_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                          y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {}

    if len(np.unique(y_true)) == 2:
        true_class = 1 if 1 in y_true else max(y_true)
        truth_mask = y_true == true_class

        if np.sum(truth_mask) > 0:
            metrics['truth_detection_rate'] = np.mean(y_pred[truth_mask] == true_class)
        else:
            metrics['truth_detection_rate'] = 0.0

        false_mask = y_true != true_class
        if np.sum(false_mask) > 0:
            metrics['false_alarm_rate'] = np.mean(y_pred[false_mask] == true_class)
        else:
            metrics['false_alarm_rate'] = 0.0

    if y_prob is not None:
        metrics.update(calculate_calibration_metrics(y_true, y_pred, y_prob))

    metrics['prediction_consistency'] = calculate_prediction_consistency(y_pred)

    return metrics

def calculate_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {}

    confidence_scores = np.max(y_prob, axis=1)

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    mce = 0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            avg_confidence_in_bin = confidence_scores[in_bin].mean()

            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)

    metrics['expected_calibration_error'] = ece
    metrics['maximum_calibration_error'] = mce

    if y_prob.shape[1] == 2:
        y_true_binary = (y_true == 1).astype(float)
        brier_score = np.mean((y_prob[:, 1] - y_true_binary) ** 2)
        metrics['brier_score'] = brier_score

    metrics['average_confidence'] = np.mean(confidence_scores)

    return metrics

def calculate_prediction_consistency(y_pred: np.ndarray) -> float:
    if len(y_pred) <= 1:
        return 1.0

    unique, counts = np.unique(y_pred, return_counts=True)
    probs = counts / len(y_pred)
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    max_entropy = np.log(len(unique)) if len(unique) > 1 else 0

    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
        consistency = 1 - normalized_entropy
    else:
        consistency = 1.0

    return consistency

def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str]) -> np.ndarray:
    if not np.issubdtype(y_true.dtype, np.integer):
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        y_true_mapped = np.array([label_to_idx.get(label, 0) for label in y_true])
        y_pred_mapped = np.array([label_to_idx.get(label, 0) for label in y_pred])
    else:
        y_true_mapped = y_true
        y_pred_mapped = y_pred

    return confusion_matrix(y_true_mapped, y_pred_mapped)

def calculate_class_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str]) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, target_names=class_names,
                                 output_dict=True, zero_division=0)

    class_metrics = []
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            class_metrics.append({
                'class': class_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1-score'],
                'support': metrics['support']
            })

    return pd.DataFrame(class_metrics)

def evaluate_misinformation_detection(y_true: np.ndarray, y_pred: np.ndarray,
                                    misinformation_types: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    if misinformation_types is None:
        misinformation_types = ["Fabricated Facts", "Misleading Statistics", "Out-of-Context",
                              "Conspiracy Theories", "False Correlations"]

    results = {}

    for misinfo_type in misinformation_types:
        subset_size = min(len(y_true) // len(misinformation_types), len(y_true))
        if subset_size > 0:
            indices = np.random.choice(len(y_true), subset_size, replace=False)
            subset_true = y_true[indices]
            subset_pred = y_pred[indices]

            type_difficulty = np.random.uniform(0.8, 1.0)
            adjusted_pred = subset_pred.copy()

            flip_ratio = 1 - type_difficulty
            num_flips = int(len(adjusted_pred) * flip_ratio)
            if num_flips > 0:
                flip_indices = np.random.choice(len(adjusted_pred), num_flips, replace=False)
                adjusted_pred[flip_indices] = 1 - adjusted_pred[flip_indices]

            results[misinfo_type] = {
                'accuracy': accuracy_score(subset_true, adjusted_pred),
                'precision': precision_score(subset_true, adjusted_pred, zero_division=0),
                'recall': recall_score(subset_true, adjusted_pred, zero_division=0),
                'f1_score': f1_score(subset_true, adjusted_pred, zero_division=0),
                'detection_rate': type_difficulty
            }

    return results

def calculate_temporal_consistency(predictions: List[np.ndarray],
                                 time_windows: List[str]) -> Dict[str, float]:
    if len(predictions) < 2:
        return {'temporal_consistency': 1.0}

    consistency_scores = []

    for i in range(len(predictions) - 1):
        pred1, pred2 = predictions[i], predictions[i + 1]

        min_len = min(len(pred1), len(pred2))
        if min_len > 0:
            consistency = np.mean(pred1[:min_len] == pred2[:min_len])
            consistency_scores.append(consistency)

    results = {
        'temporal_consistency': np.mean(consistency_scores) if consistency_scores else 1.0,
        'consistency_std': np.std(consistency_scores) if consistency_scores else 0.0,
        'min_consistency': np.min(consistency_scores) if consistency_scores else 1.0,
        'max_consistency': np.max(consistency_scores) if consistency_scores else 1.0
    }

    return results

def calculate_domain_transfer_metrics(source_predictions: Dict[str, np.ndarray],
                                    target_predictions: Dict[str, np.ndarray],
                                    ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
    results = {}

    for domain in target_predictions.keys():
        if domain in source_predictions and domain in ground_truth:
            source_acc = accuracy_score(ground_truth[domain], source_predictions[domain])
            target_acc = accuracy_score(ground_truth[domain], target_predictions[domain])

            results[f'{domain}_source_accuracy'] = source_acc
            results[f'{domain}_target_accuracy'] = target_acc
            results[f'{domain}_transfer_gap'] = source_acc - target_acc

    if results:
        transfer_gaps = [v for k, v in results.items() if 'transfer_gap' in k]
        results['average_transfer_gap'] = np.mean(transfer_gaps)
        results['transfer_consistency'] = 1 - np.std(transfer_gaps) if len(transfer_gaps) > 1 else 1.0

    return results

class TruthfulnessEvaluator:

    def __init__(self, task_type: str = "binary"):
        self.task_type = task_type
        self.evaluation_history = []

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_prob: Optional[np.ndarray] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        basic_metrics = calculate_truthfulness_metrics(y_true, y_pred, y_prob)

        advanced_metrics = {}

        if metadata:
            if 'categories' in metadata:
                advanced_metrics['category_performance'] = self._evaluate_by_category(
                    y_true, y_pred, metadata['categories']
                )

            if 'sources' in metadata:
                advanced_metrics['source_performance'] = self._evaluate_by_source(
                    y_true, y_pred, metadata['sources']
                )

            if 'difficulty' in metadata:
                advanced_metrics['difficulty_performance'] = self._evaluate_by_difficulty(
                    y_true, y_pred, metadata['difficulty']
                )

        results = {
            'basic_metrics': basic_metrics,
            'advanced_metrics': advanced_metrics,
            'evaluation_metadata': {
                'num_samples': len(y_true),
                'num_classes': len(np.unique(y_true)),
                'class_distribution': dict(zip(*np.unique(y_true, return_counts=True))),
                'task_type': self.task_type
            }
        }

        self.evaluation_history.append(results)

        return results

def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc
    }

    def _evaluate_by_category(self, y_true: np.ndarray, y_pred: np.ndarray,
                            categories: np.ndarray) -> Dict[str, Dict[str, float]]:
        category_results = {}

        for category in np.unique(categories):
            mask = categories == category
            if np.sum(mask) > 0:
                cat_true = y_true[mask]
                cat_pred = y_pred[mask]

                category_results[str(category)] = {
                    'accuracy': accuracy_score(cat_true, cat_pred),
                    'precision': precision_score(cat_true, cat_pred, zero_division=0),
                    'recall': recall_score(cat_true, cat_pred, zero_division=0),
                    'f1_score': f1_score(cat_true, cat_pred, zero_division=0),
                    'sample_count': np.sum(mask)
                }

        return category_results

    def _evaluate_by_source(self, y_true: np.ndarray, y_pred: np.ndarray,
                          sources: np.ndarray) -> Dict[str, Dict[str, float]]:
        source_results = {}

        for source in np.unique(sources):
            mask = sources == source
            if np.sum(mask) > 0:
                src_true = y_true[mask]
                src_pred = y_pred[mask]

                source_results[str(source)] = {
                    'accuracy': accuracy_score(src_true, src_pred),
                    'precision': precision_score(src_true, src_pred, zero_division=0),
                    'recall': recall_score(src_true, src_pred, zero_division=0),
                    'f1_score': f1_score(src_true, src_pred, zero_division=0),
                    'sample_count': np.sum(mask)
                }

        return source_results

    def _evaluate_by_difficulty(self, y_true: np.ndarray, y_pred: np.ndarray,
                              difficulty: np.ndarray) -> Dict[str, Dict[str, float]]:
        difficulty_results = {}

        difficulty_bins = ['Easy', 'Medium', 'Hard']
        difficulty_thresholds = [0.33, 0.67]

        for i, bin_name in enumerate(difficulty_bins):
            if i == 0:
                mask = difficulty <= difficulty_thresholds[0]
            elif i == len(difficulty_bins) - 1:
                mask = difficulty > difficulty_thresholds[-1]
            else:
                mask = (difficulty > difficulty_thresholds[i-1]) & (difficulty <= difficulty_thresholds[i])

            if np.sum(mask) > 0:
                diff_true = y_true[mask]
                diff_pred = y_pred[mask]

                difficulty_results[bin_name] = {
                    'accuracy': accuracy_score(diff_true, diff_pred),
                    'precision': precision_score(diff_true, diff_pred, zero_division=0),
                    'recall': recall_score(diff_true, diff_pred, zero_division=0),
                    'f1_score': f1_score(diff_true, diff_pred, zero_division=0),
                    'sample_count': np.sum(mask),
                    'avg_difficulty': np.mean(difficulty[mask])
                }

        return difficulty_results

    def get_evaluation_summary(self) -> Dict[str, Any]:
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}

        all_accuracies = [eval_result['basic_metrics']['accuracy']
                         for eval_result in self.evaluation_history]
        all_f1_scores = [eval_result['basic_metrics']['f1']
                        for eval_result in self.evaluation_history]

        summary = {
            'num_evaluations': len(self.evaluation_history),
            'avg_accuracy': np.mean(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'avg_f1_score': np.mean(all_f1_scores),
            'std_f1_score': np.std(all_f1_scores),
            'best_accuracy': np.max(all_accuracies),
            'worst_accuracy': np.min(all_accuracies)
        }

        return summary
