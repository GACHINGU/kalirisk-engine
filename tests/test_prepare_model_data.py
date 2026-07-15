# tests/tests_prepare_model.py

import pandas as pd
from domain.risk.prepare_model_data import prepare_model_data


def make_fake_train_test():
    train_df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "issue_d": ["Jan-2016", "Feb-2016"],
            "term": ["36 months", "60 months"],
            "grade": ["A", "B"],
            "sub_grade": ["A1", "B2"],
            "home_ownership": ["RENT", "OWN"],
            "verification_status": ["Verified", "Not Verified"],
            "purpose": ["car", "wedding"],
            "dti": [15, 20],
            "is_default": [0, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": ["3"],
            "issue_d": ["Jan-2018"],
            "term": ["36 months"],
            "grade": ["C"],
            "sub_grade": ["C1"],
            "home_ownership": ["MORTGAGE"],
            "verification_status": ["Verified"],
            "purpose": ["moving"],  # a category never seen in train, on purpose
            "dti": [25],
            "is_default": [0],
        }
    )

    return train_df, test_df


def test_test_columns_match_train_columns_exactly():
    """
    This is the critical test: even when test data contains a category
    train never saw (purpose='moving'), X_test must end up with the
    exact same columns as X_train, in the exact same order - otherwise
    the model would misread or mismatch features.
    """
    train_df, test_df = make_fake_train_test()
    X_train, X_test, y_train, y_test = prepare_model_data(train_df, test_df)

    assert list(X_train.columns) == list(X_test.columns)


def test_unseen_test_category_is_dropped():
    """Confirms 'purpose_moving' never appears, since train never saw it."""
    train_df, test_df = make_fake_train_test()
    X_train, X_test, y_train, y_test = prepare_model_data(train_df, test_df)

    assert "purpose_moving" not in X_test.columns
