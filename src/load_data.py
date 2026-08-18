"""
Data acquisition script for the CDC Diabetes Health Indicators dataset.
Source: UCI Machine Learning Repository (Dataset ID 891)
https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

def load_diabetes_data(save_raw=True, raw_path="data/raw/diabetes_raw.csv"):
    """
    Fetches the CDC Diabetes Health Indicators dataset from UCI ML Repository.
    Combines features and target into a single DataFrame.

    Parameters:
        save_raw (bool): whether to save the raw combined dataset to disk
        raw_path (str): file path to save the raw dataset

    Returns:
        pd.DataFrame: combined features + target
    """
    print("Fetching CDC Diabetes Health Indicators dataset (ID 891)...")
    dataset = fetch_ucirepo(id=891)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    if save_raw:
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"Raw dataset saved to: {raw_path}")

    return df


if __name__ == "__main__":
    df = load_diabetes_data()
    print(df.head())