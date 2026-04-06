"""
model.py

Applies XGBoost model over curated dataset to predict lead conversion.
Saves test results for further evaluation.

Usage:
    Called from main.py as part of the pipeline.
Output:
    outputs/model.joblib
"""
import os
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def run_model(input_df: pd.DataFrame, output_path: str) -> tuple:
    """
    Fit XGBoost model and predict lead conversion.
    Save the model in joblib format, and test results for further evaluation.

    Args:
        input_df: Input model dataframe
        output_path: Path to store the model file.
    Returns:
        Tuple containing model and test results.
    """
    X, y = input_df.drop('converted', axis=1), input_df['converted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)

    print(f"Test set size: {len(y_test)}")
    print(f"Predicted conversion rate: {y_pred.mean():.2%}")
    print(f"Actual conversion rate: {y_test.mean():.2%}")

    return model, y_test, y_pred
