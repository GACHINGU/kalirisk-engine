# domain/risk/explain.py

import shap
import pandas as pd


def get_feature_importance(model, top_n: int = 10) -> pd.DataFrame:
    """
    Ranks the trained model's features by how often LightGBM actually
    used them to split its trees - a global view of what the model
    relies on most, across all applicants it has ever scored. Not
    tailored to any single applicant (that's a harder, separate
    technique called SHAP, deliberately left for later).
    """
    importance_df = pd.DataFrame(
        {"feature": model.feature_name_, "importance": model.feature_importances_}
    )

    return importance_df.sort_values("importance", ascending=False).head(top_n)


def explain_applicant(model, X_new: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Explains ONE specific applicant's prediction using SHAP - unlike
    get_feature_importance() (which describes the models behavior
    globally, across every applicant it has ever scored), this answers
    "why did THIS person get THIS score" - each feature's real,
    signed contribution (positive = pushed risk up, negative = pushed
    risk down), verified to sum with the model's base score to exactly
    reproduce the real predicted probability (after a sigmoid).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_new)

    explanation_df = pd.DataFrame(
        {
            "feature": model.feature_name_,
            "shap_value": shap_values[0],
        }
    )

    explanation_df["abs_value"] = explanation_df["shap_value"].abs()

    return explanation_df.sort_values("abs_value", ascending=False).head(top_n)
