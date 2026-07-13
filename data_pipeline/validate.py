# data_pipeline/validate.py

import pandas as pd

# calling our module loader.py to hand us its function, that outputs a dictionary
from config.loader import load_settings


# A function that returns clean data and a report
# In Python, returning a tuple is the normal way to hand back more than one result.
# Small fixed bundle of values
def validate_raw_loans(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Checks every row against the valid_ranges rules in our data contract.
    Sentinel values (disguised missing data) are identified and dropped
    separately from genuinely impossible values. Rows missing a
    required_not_null field (e.g loan_amnt) are treated as non-loan
    rows entirely - such as stray footer/summary text rows found in the
    raw Lending Club export - and dropped before any other check runs.

    Returns the cleaned DataFrame, plus a report dict so we always know
    exactly what was removed and why.
    """
    # calling in the whole dictionary from settings.yaml, the data contract
    # then assigning it to a variable settings
    settings = load_settings()

    # Only focusing on the values inside the valid_ranges, which is a data_contract key
    valid_ranges = settings["data_contract"]["valid_ranges"]

    # Getting the disguised missing data (Sentinels)
    # the .get() asks data_contract to politely give the values inside missing_value_sentinels,
    # but if the box is empty or missing, don't crash the computer,
    # just give me an empty box {} instead.
    sentinels = settings["data_contract"].get("missing_value_sentinels", {})

    # Asks politely for values in require_not_null and if missing
    # hand back an empty list
    required_cols = settings["data_contract"].get("required_not_null", [])

    # Start by assuming every row is innocent (valid)
    # These creates a column of False values, one per row, same length as our data
    # Remember False acts as a place holder, assuming every value is within bounds, True for out of bound
    invalid_mask = pd.Series(False, index=df.index)
    sentinel_mask = pd.Series(False, index=df.index)
    non_loan_mask = pd.Series(False, index=df.index)
    report = {}

    # Pass 0: Identify rows that aren't real loans at all
    for column in required_cols:
        # Lets get a boolean mask colummn for null -> True and the rest False
        column_missing = df[column].isna()
        # adds all 1's correctly identified as not real loans and then appends them to the dict
        report[f"{column}_missing_count"] = int(column_missing.sum())
        # Do a comparison with the mask above and only keep True if it appears anywhere
        non_loan_mask = non_loan_mask | column_missing

    # Pass 1: identify sentinel values, column by column
    for column, bad_values in sentinels.items():
        # Ask a question, "Is these value found any where in these list (bad_values)"
        column_sentinel = df[column].isin(bad_values)
        # Adds all the 1's (correctly identified as a sentinel)
        report[f"{column}_sentinel_count"] = int(column_sentinel.sum())
        # Compare with the False placeholders using the OR operator
        # False | False are False, but the rest of the cases ar True
        sentinel_mask = sentinel_mask | column_sentinel

    """
    These for loop gives us the keys and their respective values from valid_ranges.
    It then takes the columns from the incoming dataset from path (df) and compares it 
    against the bounds. It records row to be True if bounds crossed and False if bounds
    intact. Then nicknames those boolean values column_invalid (Also sentinels are excluded). Then the column name is appended
    as  a key in the report dictionary, accompanied with sum of all the Trues it has as its respective
    value.
    """
    # Pass 2: identify genuinely out of range values, EXCLUDING sentinels
    # (a sentinel like 999 would otherwise also get counted here, double-counting it)
    for column, bounds in valid_ranges.items():
        too_low = df[column] < bounds["min"]
        too_high = df[column] > bounds["max"]

        # we use & to avoid a sentinel being double counted
        # values genuinely out of bounds and not sentinels
        # & operator favours False
        # | favours True
        column_invalid = (too_low | too_high) & (~sentinel_mask)

        report[f"{column}_invalid_count"] = int(column_invalid.sum())
        invalid_mask = invalid_mask | column_invalid

    drop_mask = invalid_mask | sentinel_mask | non_loan_mask
    clean_df = df[~drop_mask].copy()

    report["total_rows_checked"] = len(df)
    report["total_non_loan_dropped"] = int(non_loan_mask.sum())
    report["total_sentinel_dropped"] = int(sentinel_mask.sum())
    report["total_invalid_dropped"] = int(invalid_mask.sum())
    report["total_rows_dropped"] = int(drop_mask.sum())
    report["percent_dropped"] = round(100 * drop_mask.sum() / len(df), 4)

    return clean_df, report
