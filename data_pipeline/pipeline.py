# data_pipeline/pipeline.py

# The orchestrator
import pandas as pd
from data_pipeline.ingest import load_raw_loans
from data_pipeline.validate import validate_raw_loans
from data_pipeline.features import engineer_features


def run_data_pipeline(raw_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Ochestrates the full data layer: ingest -> validate -> engineer_features.

    These function does no real work itself - it simply calls the three
    already tested stages in the correct order, and returns both the
    final model-ready features AND the validation report, so data
    integrity is always visible, never silently hidden.
    """
    raw_df = load_raw_loans(raw_path)
    clean_df, report = validate_raw_loans(raw_df)
    features_df = engineer_features(clean_df)

    return features_df, report


# Nice and short, its not supposed to be clever or complicated
# Its entire value is being obviously, boringly correct.
