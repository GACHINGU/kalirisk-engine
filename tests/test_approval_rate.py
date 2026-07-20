# tests/test_approval_rate.py

import numpy as np
import pandas as pd
from domain.financial.approval import calculate_approval_rate


def make_fake_data() -> tuple[pd.Series, float]:
    """
    Make a small fake data set for the probabilities of default.
    Intentionally make the data set a pandas series.
    Then pick one cutoff point as a scalar.
    """
    pd_default = pd.Series([0.08, 0.25, 0.10, 0.21, 0.13])
    cutoff = 0.15

    return pd_default, cutoff


def test_approval_rate_calculates_expected_values() -> None:
    """
    Test whether calculate_approval_rate gives expected hand calculated
    values.
    """
    pd_default, cutoff = make_fake_data()

    approval_rate = calculate_approval_rate(pd_default, cutoff)
    expected_output = np.array([60.00])

    # we will use .isclose() since it generally checks our intent (scalar values)
    # instead of using assert  ... == which is strict on small decimal misses
    assert np.isclose(approval_rate, expected_output)
