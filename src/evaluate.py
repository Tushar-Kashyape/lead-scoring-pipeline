"""
evaluate.py

Evaluates model using test and predicted results across performance metrics.
Evaluation suite: Classification report (Precision, Recall, F1 score), AUC-ROC score
Saves results to output with execution details.

Usage:
    Called from main.py as part of the pipeline.
Output:
    outputs/evaluation_report.csv
"""
import os
import datetime

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


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

    row = {
        'timestamp': datetime.datetime.now(),
        'precision': report_dict['1']['precision'],
        'recall': report_dict['1']['recall'],
        'f1': report_dict['1']['f1-score'],
        'accuracy': report_dict['accuracy'],
        'roc_auc': roc_auc
    }

    print(f"Precision : {row['precision']:.4f}")
    print(f"Recall    : {row['recall']:.4f}")
    print(f"F1        : {row['f1']:.4f}")
    print(f"Accuracy  : {row['accuracy']:.4f}")
    print(f"ROC-AUC   : {row['roc_auc']:.4f}")

    # SHAP Explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    shap_importance = pd.DataFrame({
        "features": X_test.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by='mean_abs_shap', ascending=False)

    print("\n Top features by SHAP importance:")
    print(shap_importance.to_string(index=False))

    result = pd.DataFrame([row])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        df_results = pd.read_csv(output_path)
        df_results = pd.concat([df_results, result], ignore_index=True)
        df_results.to_csv(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
