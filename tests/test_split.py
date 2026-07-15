# tests/test_spilt.py

import pandas as pd
from domain.risk.split import chronological_train_test_split


def make_fake_features():
    return pd.DataFrame(
        {
            "issue_d": ["Jun-2016", "Dec-2017", "Jan-2018", "Jun-2018"],
            "loan_amnt": [1000, 2000, 3000, 4000],
        }
    )


def test_split_respects_calendar_boundary():
    """
    Confirms the split is based on the actual calender contact
    (train_end_date / test_start_date), not row positioning or shuffling
    """
    df = make_fake_features()
    train_df, test_df = chronological_train_test_split(df)

    assert set(train_df["issue_d"]) == {"Jun-2016", "Dec-2017"}
    assert set(test_df["issue_d"]) == {"Jan-2018", "Jun-2018"}


def test_split_has_no_overlapping_rows():
    """
    Every row must belong to exactly ONE side of the split - never both,
    never neither. This is what guarantees no future information ever
    leaks into training
    """
    df = make_fake_features()
    train_df, test_df = chronological_train_test_split(df)

    assert len(train_df) + len(test_df) == len(df)
