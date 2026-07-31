# services/training_service.py

# The training ochestrator
import os
import lightgbm as lgb
from datetime import datetime
from data_pipeline.pipeline import run_data_pipeline
from domain.risk.split import chronological_train_test_split
from domain.risk.prepare_model_data import prepare_model_data
from domain.risk.train import train_pd_model
from domain.risk.train import save_model


def train_and_save_pd_model(
    raw_path: str, model_dir: str = "data/model_artifacts"
) -> tuple[lgb.LGBMClassifier, float, dict]:
    """
    Ochestrates the full data_pipeline and the full domain.risk, in order.

    These function does no real work itself - it simply calls the tested
    data_pipeline stages from pipeline.py and then calls the tested risk
    stages in the correct order and then returns the type of model saved
    the roc-auc-score as float and the data report as a dict.
    """
    features_df, report = run_data_pipeline(raw_path)
    train_df, test_df = chronological_train_test_split(features_df)
    X_train, X_test, y_train, y_test = prepare_model_data(train_df, test_df)
    model, auc_score = train_pd_model(X_train, X_test, y_train, y_test)

    # Utilizing time-stamps, to name the path of a newly trained and saved model each time
    # the function is run
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # create this folder (and any missing parent folders) that is if it doesn't exist.
    # If it exists don't complain.
    os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/pd_model_{timestamp}.joblib"

    save_model(model, path=model_path)

    return model, auc_score, report
