# tests/test_validate.py

import pandas as pd
from data_pipeline.validate import validate_raw_loans


def make_fake_loans():
    """
    A small, fully controlled dataset - 5 rows, where we know exactly
    which ones SHOULD be dropped and why. Using real data here would
    make it impossible to know the "correct" answer in advance
    """
    return pd.DataFrame(
        {
            "dti": [20, 999, 150, -1, 30],
            "fico_range_low": [700, 700, 700, 700, 700],
            "fico_range_high": [750, 750, 750, 750, 750],
            "int_rate": [10, 10, 10, 10, 10],
        }
    )


def test_sentinel_rows_are_counted_separately_from_invalid_rows():
    df = make_fake_loans()
    clean_df, report = validate_raw_loans(df)

    assert report["dti_sentinel_count"] == 2  # the 999 and -1
    assert report["dti_invalid_count"] == 0  # 150 is real and valid, not broken


def test_genuinely_valid_high_dti_is_kept_not_dropped():
    """
    This is the most important test in these file. If this ever fails,
    it means that we're accidentally throwing away exactly the risky
    borrowers our model most needs to learn from.
    """
    df = make_fake_loans()
    clean_df, report = validate_raw_loans(df)

    assert 150 in clean_df["dti"].values
