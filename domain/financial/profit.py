# domain/financial/profit.py

import numpy as np
from config.loader import load_settings


def calculate_profit_per_loan(
    loan_amnt, int_rate, term_months, expected_loss
) -> np.ndarray | float:
    """
    The purpose of the function is to calculate profit_per_loan.
    formula: (interest_income - EL - cost_of_capital). We shall
    calculate each variable as it appears in the formulae, sequentially
    starting from interest_income.

    The function recieves expected_loss as a calculated NumPy array strictly.
    These makes sure its not dependent on calculate_expected_loss() function.
    """
    # interest_income
    # Needs: principal (loan_amnt) * int_rate (as decimal) * term_months (in years)
    # We have to make sure the variables ar actually numpy arrays
    loan_amnt = np.asarray(loan_amnt)
    int_rate = np.asarray(int_rate)
    term_months = np.asarray(term_months)

    interest_income = loan_amnt * (int_rate / 100) * (term_months / 12)

    # expected_loss
    # making sure its a numpy array
    expected_loss = np.asarray(expected_loss)

    # cost_of_capital
    # Needs: capital_tied(loan_amnt) * cost_of_capital_rate(as decimal) * term_months (in years)
    # 1st task: get the cost_of_capital_rate from the data contract
    settings = load_settings()
    cost_of_capital_rate = settings["data_contract"]["financial_assumptions"][
        "cost_of_capital_rate"
    ]

    cost_of_capital = loan_amnt * (cost_of_capital_rate / 100) * (term_months / 12)

    # calculating profit_per_loan from the derived variables

    profit_per_loan = interest_income - expected_loss - cost_of_capital

    return profit_per_loan
