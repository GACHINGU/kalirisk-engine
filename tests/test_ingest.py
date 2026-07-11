# tests/test_ingest.py

from data_pipeline.ingest import load_raw_loans

RAW_PATH = "data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"


def test_load_raw_loans_has_expected_columns():
    """
    Confirm our ingestion only returns our approve gold columns,
    nothing extra.
    """

    df = load_raw_loans(RAW_PATH)

    expected_columns = {
        "id",
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "grade",
        "sub_grade",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "issue_d",
        "loan_status",
        "purpose",
        "dti",
        "fico_range_low",
        "fico_range_high",
    }

    # A set is a collection where order doesnt matter and duplicates aren't allowed
    # Perfect for do these 2 groups contain exactly the same things
    assert set(df.columns) == expected_columns


def test_load_raw_loans_has_rows():
    """
    A basic sanity check - we should never silently get an empty dataframe.
    """

    df = load_raw_loans(RAW_PATH)

    assert len(df) > 0
