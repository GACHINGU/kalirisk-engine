# tests/test_optimizer.py

import numpy as np
import pandas as pd
from domain.portfolio.optimizer import find_optimal_cutoff


def make_fake_test_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Make a fake test data, intentionally pandas Series to see whether the to NumPy
    arrays conversion happens.

    Also an excessively high el to cross max el ratio bounds to see whether the
    constraint works.
    """
    pd_default = pd.Series([0.02, 0.04, 0.13, 0.16, 0.17])
    profit_per_loan = pd.Series([200, 300, 150, 250, 350])
    expected_loss = pd.Series([53, 100, 64, 72, 300])
    loan_amnt = pd.Series([1000, 1300, 900, 1200, 1500])

    return pd_default, profit_per_loan, expected_loss, loan_amnt


def test_find_optimal_cutoff_returns_numpyscalars() -> None:
    """
    Test whether the output tuple (best_cutoff, best_profit) is truly a NumPy array.
    """
    pd_default, profit_per_loan, expected_loss, loan_amnt = make_fake_test_data()

    best_cutoff, best_profit = find_optimal_cutoff(
        pd_default, profit_per_loan, expected_loss, loan_amnt
    )

    # checking whether best_cutoff, best_profit are trully numpy arrays
    assert np.isscalar(best_cutoff)
    assert np.isscalar(best_profit)


def test_find_optimal_cutoff_gives_expected_values() -> None:
    """
    Testing whether the find_optimal_cutoff gives the expected outputs.
    """
    pd_default, profit_per_loan, expected_loss, loan_amnt = make_fake_test_data()
    best_cutoff, best_profit = find_optimal_cutoff(
        pd_default, profit_per_loan, expected_loss, loan_amnt
    )

    assert np.isclose(best_cutoff, 0.17)
    assert np.isclose(best_profit, 900)
