# tests/test_model_persistence.py

import numpy as np
import pandas as pd
import lightgbm as lgb

from domain.risk.train import save_model, load_model


def make_tiny_training_data():
    """A small, fast fake dataset - enough for LightGBM to actually train on."""
    X = pd.DataFrame(
        {
            "dti": [15, 40, 22, 55, 18, 60, 25, 45],
            "fico_range_low": [750, 620, 700, 600, 780, 590, 710, 640],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    return X, y


def test_saved_and_loaded_model_give_identical_predictions(tmp_path):
    """
    If saved/loaded truly preserves everything the model learned, predictions
    from the original model and the reloaded model must be EXACTLY identical
    - not just similar, not just close, but the same numbers.
    """
    X, y = make_tiny_training_data()

    model = lgb.LGBMClassifier(
        n_estimators=10, max_depth=2, random_state=42, verbose=-1
    )
    model.fit(X, y)

    save_path = tmp_path / "test_model.joblib"
    save_model(model, path=str(save_path))
    loaded_model = load_model(path=str(save_path))

    original_predictions = model.predict_proba(X)[:, 1]
    loaded_predictions = loaded_model.predict_proba(X)[:, 1]

    assert np.array_equal(original_predictions, loaded_predictions)
