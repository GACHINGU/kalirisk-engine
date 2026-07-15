# domain/risk/split.py

import pandas as pd
from config.loader import load_settings


def chronological_train_test_split(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits features into train/test using a fixed calender boundary,
    not a raw-count percentage - avoiding both future-leakage (shuffling)
    and arbitrary mid-month ties (row-based cutoffs).
    """
    # Call in the data_contract from config/settings.yaml, containing the train_test agreement.
    settings = load_settings()
    split_cfg = settings["data_contract"]["train_test_split"]

    # Change the values in train_test_split into actual date times
    train_end = pd.to_datetime(split_cfg["train_end_date"])
    test_start = pd.to_datetime(split_cfg["test_start_date"])

    # Change the incoming main dataframe column 'issue_d' to also actual dates
    issue_dates = pd.to_datetime(features_df["issue_d"], format="%b-%Y")

    # Split the data into independent train and test tables, respectively
    train_df = features_df[issue_dates <= train_end].copy()
    test_df = features_df[issue_dates >= test_start].copy()

    return train_df, test_df
