# domain/financial/approval.py

import numpy as np


def calculate_approval_rate(pd_default, cutoff) -> float:
    """
    Problem statement: These function sole purpose is to calculates approval rate from
    cut off points and probabilities of defaults given. It will be sum of loans approved
    over the total applicants multplied by a 100 (since its a rate). Every variable must
    pass the NumPy array sanity check before calculating approval.

    Disclaimer: No DataFrame, web code, model code in these function.
    Its strictly a quantitative finance function.
    """
    # enforce a numpy array on pd_default
    pd_default = np.asarray(pd_default)

    # enforce a numpy array on cutoff
    cutoff = np.asarray(cutoff)

    # create an approval mask
    approval_mask = pd_default < cutoff

    # calculate approval
    approval_rate = approval_mask.sum() / len(pd_default) * 100

    return approval_rate
