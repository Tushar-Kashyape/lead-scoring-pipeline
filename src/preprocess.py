"""
preprocess.py

Identifies and removes duplicate lead entries from raw data.
Deduplication key: phone_number.

Usage:
    Called from main.py as part of the pipeline.
Output:
    data/processed/leads_clean.csv
"""

import os
import pandas as pd

def run_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Remove duplicate entries from raw data based on keys which don't change
    after creation. Save the updated clean dataframe separately.
    Deduplication key: phone_number.

    Args:
        input_path (str): Path to raw data file
        output_path (str): Path to clean data file

    Returns:
        Clean dataframe with duplicate entries removed.
    """
    df = pd.read_csv(input_path)
    pre_dedupe_count = df.shape[0]
    df.drop_duplicates(subset=['phone_number'], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    dup_count = pre_dedupe_count - df.shape[0]

    print(f"No. of duplicates: {dup_count}")
    print(f"No. of rows remaining: {df.shape[0]}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
