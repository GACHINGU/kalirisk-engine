# domain/risk/prepare_model_data.py

import pandas as pd

CATEGORICAL_COLS = ["sub_grade", "home_ownership", "verification_status", "purpose"]
DROP_COLS = ["id", "issue_d", "term", "grade", "is_default"]


def prepare_model_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Converts train/tests DataFrames into model-ready X/y arrays.
    One-hot encodes categorical columns, then forces tests to match
    train's exact column shape - preventing silent misalignment between
    the two.
    """
    X_train = train_df.drop(columns=DROP_COLS)
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS)
    y_train = train_df["is_default"]

    # making sure the exam aligns too
    X_test = test_df.drop(columns=DROP_COLS)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = test_df["is_default"]

    return X_train, X_test, y_train, y_test
