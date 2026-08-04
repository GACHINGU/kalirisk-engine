# tests/ test_decisioning_service.py

import numpy as np
import pandas as pd
from tests.test_training_service import make_fake_raw_csv
from services.decisioning_service import decide_best_cutoff_and_profit


# creating a fake model that actually learns nothing
class FakeModel:
    def __init__(self, feature_name_):
        self.feature_name_ = feature_name_

    def predict_proba(self, X):
        n_rows = len(X)
        fake_probs_of_default = np.linspace(0.05, 0.45, n_rows)
        fake_probs_of_no_default = 1 - fake_probs_of_default
        return np.column_stack([fake_probs_of_no_default, fake_probs_of_default])


def test_decide_best_cutoff_and_profit_return_correct_structure(tmp_path) -> None:
    """
    Confirms decide_best_cutoff_and_profit() correctly wires validation,
    feature engineering, model alignment, financial math, and the
    optimizer together - using a FakeModel that never learns anything,
    since we are testing the WIRING here, not prediction accuracy
    (already proven separately by test_optimizer.py and friends)
    """
    # first we need a pandas DataFrame
    fake_csv = tmp_path / "scratch_fake_loans.csv"
    make_fake_raw_csv(str(fake_csv), n_rows=50)

    new_applicants = pd.read_csv(str(fake_csv))

    # second we need a FakeModel

    fake_feature_names = [
        "loan_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "term_months",
        "grade_numeric",
        "installment_to_income_ratio",
        "sub_grade_A1",
        "sub_grade_B2",
        "sub_grade_C3",
        "sub_grade_D4",
        "home_ownership_MORTGAGE",
        "home_ownership_OWN",
        "home_ownership_RENT",
        "verification_status_Not Verified",
        "verification_status_Verified",
        "purpose_car",
        "purpose_credit_card",
        "purpose_debt_consolidation",
    ]

    fake_model = FakeModel(feature_name_=fake_feature_names)

    best_cutoff, best_profit, approval_rate, full_report = (
        decide_best_cutoff_and_profit(df=new_applicants, model=fake_model)
    )

    # check if truly the full_report is a dictionary
    assert isinstance(full_report, dict)

    # check if data_quality_report is in the full report
    assert "data_quality_report" in full_report

    # check if approval_rate is in the full report
    assert "approval_rate" in full_report

    # check if trulu num_applicants_evaluated is 50 as per the hardcoded n_rows
    assert full_report["num_applicants_evaluated"] == 50
