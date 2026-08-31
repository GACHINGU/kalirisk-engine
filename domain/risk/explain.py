# domain/risk/explain.py

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
