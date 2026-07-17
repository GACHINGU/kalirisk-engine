# domain/financial/expected_loss.py

import numpy as np
from config.loader import load_settings


def calculate_expected_loss(loan_amnt, pd_default):
    """
    These function sole purpose is to calculate Expected Loss (EL).
    EL = pd_default * loan_amnt * lgd.

    But first we have to make sure the data we are ingesting is purely
    numpy arrays, to prevent index mismatch

    Strictly financial mathematics, no model or web code.
    """
    # getting lgd from the data contract
    settings = load_settings()
    lgd = settings["data_contract"]["financial_assumptions"]["lgd"]

    # loan_amnt if you are a pandas series, come in,
    # If you are already an array you can also come in
    loan_amnt = np.asarray(loan_amnt)

    # pd_default if you are a pandas series, come in
    # If already a numpy array also come in
    pd_default = np.asarray(pd_default)

    # calculating Expected Loss (EL)
    expected_loss = pd_default * loan_amnt * lgd

    return expected_loss
