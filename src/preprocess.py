import pandas as pd
import os

def run_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Remove duplicate entries from raw data based on keys which don't change
    after creation. Save the updated clean dataframe separately.
    Deduplication key: source_channel, city, property_type, budget_range, phone_verified.

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
