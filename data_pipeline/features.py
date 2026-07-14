# data_pipeline/features.py

import pandas as pd
from config.loader import load_settings


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms validated raw loans data into model-ready numeric features.
    Builds the target column (is_default) from loan_status, then DROPS
    laon_status immediatley afterwards, so it never leaks into the model
    as a feature - enforcing the leakage_excluded rule from our contract.
    """
    settings = load_settings()
    df = df.copy()

    # --- term: "36 months" -> 36.0 ---
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)

    # --- grade: "A"..."G" -> 1...7, ordered by increasing risk ---
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade_numeric"] = df["grade"].map(grade_map)

    # --- installment_to_income_ratio, with zero_income guard ---
    monthly_income = df["annual_inc"] / 12
    df["installment_to_income_ratio"] = df["installment"] / (monthly_income + 1e-5)

    # --- target: is_default, built from loan_status using our contract ---
    target_cfg = settings["data_contract"]["target"]
    source_col = target_cfg["source_column"]
    bad_statuses = target_cfg["default_statuses"]
    output_col = target_cfg["output_column"]

    df[output_col] = df[source_col].isin(bad_statuses).astype(int)

    # --- enforce leakage rule: drop every column listed in leakage_excluded ---
    leakage_cols = settings["data_contract"]["leakage_excluded"]
    df = df.drop(columns=leakage_cols, errors="ignore")

    return df
