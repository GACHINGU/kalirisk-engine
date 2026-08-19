# domain/risk/prepare_new_applicant_data.py

import pandas as pd
from domain.risk.prepare_model_data import CATEGORICAL_COLS, DROP_COLS


def prepare_new_applicant_data(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Prepares brand new, never-before-seen applicant data for prediction.

    Unlike prepare_model_data() (which aligns X_test against X_train's
    columns during training/evaluation), these function has no train_df
    to align against - a live applicant is not a part of any train/test
    split. Instead, it aligns against the trained model's OWN memory of
    which columns it expects (model.feature_name_), since that is the
    single source of truth for what the model can actually use.
    """
    X_new = df.drop(columns=DROP_COLS, errors="ignore")
    X_new["verification_status"] = X_new["verification_status"].str.replace(" ", "_")
    X_new = pd.get_dummies(X_new, columns=CATEGORICAL_COLS)
    X_new = X_new.reindex(columns=model.feature_name_, fill_value=0)

    return X_new
