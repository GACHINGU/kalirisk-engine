# tests/test_prepare_new_applicant_data.py

import pandas as pd
import lightgbm as lgb
from types import SimpleNamespace
from domain.risk.prepare_new_applicant_data import prepare_new_applicant_data


def make_fake_model_feature_names_and_fake_new_applicant_data() -> tuple[
    lgb.LGBMClassifier, pd.DataFrame
]:
    """
    These function mimicks the models OWN memory, the saved feature_names as list.
    Also mimicks a new applicants dataframe.
    """
    model = SimpleNamespace(
        feature_name_=[
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
            "sub_grade_A2",
            "sub_grade_A3",
            "sub_grade_A4",
            "sub_grade_A5",
            "sub_grade_B1",
            "sub_grade_B2",
            "sub_grade_B3",
            "sub_grade_B4",
            "sub_grade_B5",
            "sub_grade_C1",
            "sub_grade_C2",
            "sub_grade_C3",
            "sub_grade_C4",
            "sub_grade_C5",
            "sub_grade_D1",
            "sub_grade_D2",
            "sub_grade_D3",
            "sub_grade_D4",
            "sub_grade_D5",
            "sub_grade_E1",
            "sub_grade_E2",
            "sub_grade_E3",
            "sub_grade_E4",
            "sub_grade_E5",
            "sub_grade_F1",
            "sub_grade_F2",
            "sub_grade_F3",
            "sub_grade_F4",
            "sub_grade_F5",
            "sub_grade_G1",
            "sub_grade_G2",
            "sub_grade_G3",
            "sub_grade_G4",
            "sub_grade_G5",
            "home_ownership_ANY",
            "home_ownership_MORTGAGE",
            "home_ownership_NONE",
            "home_ownership_OTHER",
            "home_ownership_OWN",
            "home_ownership_RENT",
            "verification_status_Not_Verified",
            "verification_status_Source_Verified",
            "verification_status_Verified",
            "purpose_car",
            "purpose_credit_card",
            "purpose_debt_consolidation",
            "purpose_educational",
            "purpose_home_improvement",
            "purpose_house",
            "purpose_major_purchase",
            "purpose_medical",
            "purpose_moving",
            "purpose_other",
            "purpose_renewable_energy",
            "purpose_small_business",
            "purpose_vacation",
            "purpose_wedding",
        ]
    )
    new_applicant_data = pd.DataFrame(
        {
            "id": ["3"],
            "issue_d": ["Jan-2018"],
            "term": ["36 months"],
            "grade": ["C"],
            "sub_grade": ["C1"],
            "home_ownership": ["MORTGAGE"],
            "verification_status": ["Verified"],
            "purpose": ["moving"],  # a category never seen by the model, on purpose
            "dti": [25],
        }
    )

    return new_applicant_data, model


def test_new_applicant_data_columns_match_model_feature_data_exactly() -> None:
    """
    These is the critical test: even when new_applicant_data contains a category
    the model never saw (purpose= "moving"), or new_applicant_data has a missing
    column ("is_default"),new_applicant_data must end up
    with the same exact columns the model memorized, in the exact same order.
    Otherwise there maybe a mismatch during prediction.
    """
    new_applicant_data, model = (
        make_fake_model_feature_names_and_fake_new_applicant_data()
    )
    new_applicant_data = prepare_new_applicant_data(new_applicant_data, model)

    assert model.feature_name_ == list(new_applicant_data)
