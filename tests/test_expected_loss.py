# tests/test_expected_loss.py

import numpy as np
import pandas as pd
from domain.financial.expected_loss import calculate_expected_loss


def make_fake_loan_amnt_and_pd_default_data() -> tuple[pd.Series, pd.Series]:
    """We intentionally make the loan_amnt and pd_default data pandas series."""
    # loan_amnt as a pandas series
    loan_amnt = pd.Series([10000, 20000, 5000])

    # pd_default as a pandas series
    pd_default = pd.Series([0.10, 0.20, 0.50])

    return loan_amnt, pd_default


def test_expected_loss_is_numpyarray() -> None:
    """We test whether the returned expected_loss is NumPy array."""
    loan_amnt, pd_default = make_fake_loan_amnt_and_pd_default_data()

    expected_loss = calculate_expected_loss(loan_amnt, pd_default)

    # checking whether truly expected_loss is a NumPy array
    assert isinstance(expected_loss, np.ndarray)


def test_expected_loss_calculates_correct_values() -> None:
    """We test whether multplication occurs, and that  it occurs in the right order."""
    loan_amnt, pd_default = make_fake_loan_amnt_and_pd_default_data()

    expected_loss = calculate_expected_loss(loan_amnt, pd_default)
    expected_output = np.array([900, 3600, 2250])

    # check whether multiplication actually occurs, and occurs in the right order
    # I will use a strict teacher who doesn't care if you're off by 0.000000000001

    assert np.array_equal(expected_loss, expected_output)
