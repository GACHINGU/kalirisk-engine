# tests/test_profit_per_loan.py

import pandas as pd
import numpy as np
from domain.financial.profit import calculate_profit_per_loan


def make_fake_test_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Making fake test data, intentionally pd.Series to check whether the conversions
    happen.
    """
    loan_amnt = pd.Series([10000, 15000, 20000])
    int_rate = pd.Series([12, 16, 18])  # as percentage not decimal
    term_months = pd.Series([36, 42, 48])
    expected_loss = pd.Series([900, 1100, 1300])

    return loan_amnt, int_rate, term_months, expected_loss


def test_profit_per_loan_is_numpyarray() -> None:
    """
    Testing whether the output (profit_per_loan) is trully a numpy array.
    """
    loan_amnt, int_rate, term_months, expected_loss = make_fake_test_data()

    profit_per_loan = calculate_profit_per_loan(
        loan_amnt, int_rate, term_months, expected_loss
    )

    # checking whether trully, profit_per_loan is a numpy array
    assert isinstance(profit_per_loan, np.ndarray)


def test_profit_per_loan_calculates_expected_values() -> None:
    """
    Testing the calculate_profit_per_loan() function calculates the expected output.
    """
    loan_amnt, int_rate, term_months, expected_loss = make_fake_test_data()

    profit_per_loan = calculate_profit_per_loan(
        loan_amnt, int_rate, term_months, expected_loss
    )
    expected_output = np.array([1200, 4675, 9100])

    assert np.allclose(profit_per_loan, expected_output)
