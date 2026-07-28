# domain/portfolio/optimizer.py

import numpy as np
from config.loader import load_settings


def find_optimal_cutoff(
    pd_default, profit_per_loan, expected_loss, loan_amnt
) -> tuple[np.float64, np.float64]:
    """
    These functions seeks to find the best cutoff and the best profit within the
    hard set maximum expected loss ratio constraint.

    Its strictly a quantitative mathmatics function, no model or web code.
    """
    # calling in the maximum el constraint
    settings = load_settings()
    max_el_ratio = settings["data_contract"]["financial_assumptions"]["max_el_ratio"]

    # making sure that all the input variables are numpy arrays
    pd_default = np.asarray(pd_default)
    profit_per_loan = np.asarray(profit_per_loan)
    expected_loss = np.asarray(expected_loss)
    loan_amnt = np.asarray(loan_amnt)

    # creating candidate cutoffs
    candidate_cutoffs = np.arange(0.01, 0.51, 0.01)

    best_cutoff = None
    best_profit = -np.inf  # incase of negative profits (losses)

    # looping to find the best cutoff and best profit within constraint
    for cutoff in candidate_cutoffs:
        approved_mask = pd_default < cutoff
        total_profit = profit_per_loan[approved_mask].sum()
        total_el = expected_loss[approved_mask].sum()
        total_amnt = loan_amnt[approved_mask].sum()
        el_ratio = total_el / total_amnt if total_amnt > 0 else 0

        # utilizing the hard constraint
        if el_ratio <= max_el_ratio and total_profit > best_profit:
            best_profit = total_profit
            best_cutoff = cutoff

    return best_cutoff, best_profit
