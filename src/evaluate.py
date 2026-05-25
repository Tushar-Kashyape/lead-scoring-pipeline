"""
evaluate.py

Evaluates model using test and predicted results across performance metrics.
Evaluation suite: Classification report (Precision, Recall, F1 score), AUC-ROC score,
SHAP feature importance.Saves results to output with execution details.

Usage:
    Called from main.py as part of the pipeline.
Output:
    outputs/evaluation_report.csv
    outputs/shap_importance.csv
"""
import os
import datetime

import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, \
    precision_recall_curve, f1_score

SHAP_OUTPUT_PATH = 'outputs/shap_importance.csv'
RECALL_THRESHOLD = 0.85


def run_evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series,
                 y_pred: np.ndarray, output_path: str) -> None:
    """
    Evaluate model and results over evaluation suite.
    Save the results to outputs with execution details.

    Args:
        model: XGBoost classifier.
        X_test: DataFrame of test features.
        y_test: Pandas Series of actual labels.
        y_pred: Numpy array of predicted labels.
        output_path: Path to save evaluation report.
    Returns:
        None
    """
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Threshold tuning - optimal, constrained
    f1_threshold, recall_threshold = tune_threshold(model, X_test, y_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_f1 = (y_pred_proba >= f1_threshold).astype(int)
    y_pred_recall = (y_pred_proba >= recall_threshold).astype(int)

    report_dict_f1 = classification_report(y_test, y_pred_f1, output_dict=True)
    report_dict_recall = classification_report(y_test, y_pred_recall,
                                               output_dict=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    rows = [
        {
            "run_id": run_id,
            "threshold_type": "default",
            "threshold": 0.5,
            "precision": round(report_dict["1"]["precision"], 4),
            "recall": round(report_dict["1"]["recall"], 4),
            "f1_score": round(report_dict["1"]["f1-score"], 4),
            "accuracy": round(report_dict["accuracy"], 4),
            "roc_auc": round(roc_auc, 4),
        },
        {
            "run_id": run_id,
            "threshold_type": "f1_optimal",
            "threshold": round(float(f1_threshold), 4),
            "precision": round(report_dict_f1["1"]["precision"], 4),
            "recall": round(report_dict_f1["1"]["recall"], 4),
            "f1_score": round(report_dict_f1["1"]["f1-score"], 4),
            "accuracy": round(report_dict_f1["accuracy"], 4),
            "roc_auc": round(roc_auc, 4),
        },
        {
            "run_id": run_id,
            "threshold_type": "recall_constrained",
            "threshold": round(float(recall_threshold), 4),
            "precision": round(report_dict_recall["1"]["precision"], 4),
            "recall": round(report_dict_recall["1"]["recall"], 4),
            "f1_score": round(report_dict_recall["1"]["f1-score"], 4),
            "accuracy": round(report_dict_recall["accuracy"], 4),
            "roc_auc": round(roc_auc, 4),
        }
    ]

    for r in rows:
        print(f"\nThreshold: {r['threshold_type']} ({r['threshold']})")
        print(f"Precision : {r['precision']} | Recall : {r['recall']} | "
              f"F1 : {r['f1_score']} | Accuracy : {r['accuracy']} | ROC-AUC : {r['roc_auc']}")

    results = pd.DataFrame(rows)
    save_results(output_path, results)

    # SHAP Explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    shap_importance = pd.DataFrame({
        "features": X_test.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by='mean_abs_shap', ascending=False)

    print("\n Top features by SHAP importance:")
    print(shap_importance.to_string(index=False))

    shap_row = {'timestamp': datetime.datetime.now()}
    shap_row.update(
        shap_importance.set_index('features')['mean_abs_shap'].to_dict()
    )

    shap_result = pd.DataFrame([shap_row])
    save_results(SHAP_OUTPUT_PATH, shap_result)


def tune_threshold(model: XGBClassifier, X_test: pd.DataFrame,
                   y_test: pd.Series) -> tuple:
    """
    Find the optimal threshold by plotting the precision-recall curve and picking
    the point that best serves the business.

    Args:
        model: XGBoost classifier.
        X_test: DataFrame of test features.
        y_test: Pandas Series of actual labels.
    Returns:
        Optimal model threshold, recall constrained threshold.
    """

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test,
                                                             y_pred_proba)

    # Drop last element — sklearn adds a boundary point with no corresponding threshold
    f1_scores = (2 * (precisions[:-1] * recalls[:-1]) /
                 (precisions[:-1] + recalls[:-1]))
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    # Among all thresholds where we catch 85%+ of converters, pick the one with the highest precision.
    recall_constrained_idx = np.where(recalls[:-1] >= RECALL_THRESHOLD)[0]
    recall_constrained_threshold = thresholds[
        recall_constrained_idx[np.argmax(precisions[recall_constrained_idx])]]

    print(f"Optimal threshold (F1-based)           : {optimal_threshold:.4f}")
    print(f"Optimal threshold (Recall-constrained) :"
          f" {recall_constrained_threshold:.4f}")

    return optimal_threshold, recall_constrained_threshold


def save_results(path: str, output_df: pd.DataFrame) -> None:
    """
    Append output DataFrame to CSV at given path.
    Creates the file if it doesn't exist.

    Args:
        path: Path to save CSV file.
        output_df: DataFrame to append.
    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        df_shap = pd.read_csv(path)
        output_df = pd.concat([df_shap, output_df], ignore_index=True)

    output_df.to_csv(path, index=False)
