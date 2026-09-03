# tests/test_explain.py

import shap
import numpy as np
import pandas as pd
import lightgbm as lgb
from domain.risk.explain import get_feature_importance
from domain.risk.explain import explain_applicant


# creating a fake model, for testing
class FakeModelForImportance:
    def __init__(self, feature_name_, feature_importances_):
        self.feature_name_ = feature_name_
        self.feature_importances_ = feature_importances_


def test_get_feature_importance_sorts_correctly() -> None:
    """
    Confirm get_feature_importance() correctly sorts features by
    importance, highest first - using a tiny, fully known fake model
    instead of the real 68-feature trained model.
    """
    fake_model = FakeModelForImportance(
        feature_name_=["dti", "fico_range_low", "loan_amnt"],
        feature_importances_=[50, 200, 10],
    )

    result = get_feature_importance(fake_model, top_n=3)

    assert result.iloc[0]["feature"] == "fico_range_low"
    assert result.iloc[0]["importance"] == 200


def test_explain_applicant_sums_to_real_prediction() -> None:
    """
    Confirms explain_applicant() produces SHAP values that genuinely
    sum (with the model's base score, through a sigmoid) to the real
    predicted probability - proving the explanation is mathematically
    honest, not just plausible-looking numbers.
    """
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "dti": np.random.uniform(5, 40, 50),
            "fico_range_low": np.random.randint(600, 800, 50),
        }
    )
    y = pd.Series(np.random.randint(0, 2, 50))

    model = lgb.LGBMClassifier(
        n_estimators=10, max_depth=2, random_state=42, verbose=-1
    )
    model.fit(X, y)

    one_applicant = X.iloc[[0]]

    explanation_df = explain_applicant(model, one_applicant, top_n=2)

    explainer = shap.TreeExplainer(model)
    raw_score = explainer.expected_value + explanation_df["shap_value"].sum()
    reconstructed_prob = 1 / (1 + np.exp(-raw_score))

    real_prob = model.predict_proba(one_applicant)[:, 1][0]

    assert np.isclose(reconstructed_prob, real_prob)
