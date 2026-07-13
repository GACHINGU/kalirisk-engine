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
            "loan_amnt": [5000, 5000, 5000, 5000, 5000],
        }
    )


def test_sentinel_rows_are_counted_separately_from_invalid_rows():
    df = make_fake_loans()
    clean_df, report = validate_raw_loans(df)

    assert report["dti_sentinel_count"] == 2  # the 999 and -1
    assert report["dti_invalid_count"] == 0  # 150 is real and valid, not broken


def test_rows_missing_loan_amnt_are_dropped_as_non_loan_rows():
    """
    Real Lending Club exports contain stray footer/summary text rows
    (e.g "Total amount funded in policy code ..."), which land in the
    dataframe with avery real column empty. These aren't loans and
    must never reach feature engineering or the model.
    """
    # creating a fake dataframe
    df = pd.DataFrame(
        {
            "dti": [20, 30],
            "fico_range_low": [700, 700],
            "fico_range_high": [750, 750],
            "int_rate": [10, 10],
            "loan_amnt": [5000, None],  # second row mimics a footer row
        }
    )

    clean_df, report = validate_raw_loans(df)

    assert report["loan_amnt_missing_count"] == 1
    assert len(clean_df) == 1


def test_genuinely_valid_high_dti_is_kept_not_dropped():
    """
    This is the most important test in these file. If this ever fails,
    it means that we're accidentally throwing away exactly the risky
    borrowers our model most needs to learn from.
    """
    df = make_fake_loans()
    clean_df, report = validate_raw_loans(df)

    # Since 150 is high, but genuinly valid, within the set bounds in settings.yaml
    assert 150 in clean_df["dti"].values
