# services/decisioning_service.py

import pandas as pd
import numpy as np

from data_pipeline.validate import validate_raw_loans
from data_pipeline.features import engineer_features
from domain.risk.prepare_new_applicant_data import prepare_new_applicant_data
from domain.financial.expected_loss import calculate_expected_loss
from domain.financial.profit import calculate_profit_per_loan
from domain.financial.approval import calculate_approval_rate
from domain.portfolio.optimizer import find_optimal_cutoff


def decide_best_cutoff_and_profit(
    df: pd.DataFrame, model
) -> tuple[np.float64, np.float64, np.float64, dict]:
    """
    Takes in new applicant data and an already-loaded model. Validates
    and engineers features, aligns to the models expected columns,
    predicts defaults probability, then runs the full financial engine
    and optimizer to produce a real lending decision.

    Returns (best_cutoff, best_profit, approval_rate, report), where
    report separates data-quality findings from the decision itself,
    so a caller never has to guess which is which.
    """
    clean_df, report = validate_raw_loans(df)
    df = engineer_features(clean_df)

    X_new = prepare_new_applicant_data(df, model)
    pd_default = model.predict_proba(X_new)[:, 1]

    loan_amnt = df["loan_amnt"]
    int_rate = df["int_rate"]
    term_months = df["term_months"]

    expected_loss = calculate_expected_loss(loan_amnt=loan_amnt, pd_default=pd_default)
    profit_per_loan = calculate_profit_per_loan(
        loan_amnt=loan_amnt,
        int_rate=int_rate,
        term_months=term_months,
        expected_loss=expected_loss,
    )
    best_cutoff, best_profit = find_optimal_cutoff(
        pd_default=pd_default,
        profit_per_loan=profit_per_loan,
        expected_loss=expected_loss,
        loan_amnt=loan_amnt,
    )
    approval_rate = calculate_approval_rate(pd_default=pd_default, cutoff=best_cutoff)

    full_report = {
        "data_quality_report": report,
        "approval_rate": approval_rate,
        "num_applicants_evaluated": len(df),
    }

    return best_cutoff, best_profit, approval_rate, full_report
