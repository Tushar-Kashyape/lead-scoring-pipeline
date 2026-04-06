"""
features.py

Module to transform cleaned data into suitable format for the model.
Transforms categorical variable string values in numerical values,
drops identifiers.

Usage:
    Called from main.py as part of the pipeline.
Output:
    data/processed/leads_model.csv
"""
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def run_feature_engineering(input_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Transforms raw categorical strings into numbers and drop identifiers.

    Args:
        input_df: input clean dataframe
        output_path: Path to model-suitable data file

    Returns:
        Transformed dataframe.
    """
    df = input_df.drop(columns=['lead_id', 'phone_number'])
    cat_col_list = ['source_channel', 'city', 'property_type', 'budget_range']

    le = LabelEncoder()
    for col in cat_col_list:
        df[col] = le.fit_transform(df[col])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Feature engineering complete. Shape: {df.shape}")
    return df
