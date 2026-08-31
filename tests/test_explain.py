# tests/test_explain.py

from domain.risk.explain import get_feature_importance


# creating a fake model, for testing
class FakeModelForImportance:
    def __init__(self, feature_name_, feature_importances_):
        self.feature_name_ = feature_name_
        self.feature_importances_ = feature_importances_


def test_get_feature_importance_sorts_correctly() -> None:
    """
    Confirm get_feature_importance() correctly sorts features by
    importanc, highest first - using a tiny, fully known fake model
    instead of the real 68-feature trained model.
    """
    fake_model = FakeModelForImportance(
        feature_name_=["dti", "fico_range_low", "loan_amnt"],
        feature_importances_=[50, 200, 10],
    )

    result = get_feature_importance(fake_model, top_n=3)

    assert result.iloc[0]["feature"] == "fico_range_low"
    assert result.iloc[0]["importance"] == 200
