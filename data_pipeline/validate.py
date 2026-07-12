# data_pipeline/validate.py

import pandas as pd

# calling our module loader.py to hand us its function, that outputs a dictionary
from config.loader import load_settings


# A function that returns clean data and a report
# In Python, returning a tuple is the normal way to hand back more than one result.
# Small fixed bundle of values
def validate_raw_loans(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Checks every row against the valid_ranges in our data contract.
    Sentinel values (disguised missing values) are identified and
    dropped separately from genuinly impossible values, so our
    report tells an honest story about why each row was removed.

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

    # Start by assuming every row is innocent (valid)
    # These creates a column of False values, one per row, same length as our data
    # Remember False acts as a place holder, assuming every value is within bounds, True for out of bound
    invalid_mask = pd.Series(False, index=df.index)
    sentinel_mask = pd.Series(False, index=df.index)
    report = {}

    # First pass: identify sentinel values, column by column
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
    # Second pass: identify genuinely out of range values, EXCLUDING sentinels
    # (a sentinel like 999 would otherwise also get counted here, double-counting it)
    for column, bounds in valid_ranges.items():
        too_low = df[column] < bounds["min"]
        too_high = df[column] > bounds["max"]

        # we use & to avoid a sentinel being double counted
        # values genuinely out of bounds and not sentinels
        column_invalid = (too_low | too_high) & (~sentinel_mask)

        report[f"{column}_invalid_count"] = int(column_invalid.sum())
        invalid_mask = invalid_mask | column_invalid

    drop_mask = invalid_mask | sentinel_mask
    clean_df = df[~drop_mask].copy()

    report["total_rows_checked"] = len(df)
    report["total_sentinel_dropped"] = int(sentinel_mask.sum())
    report["total_invalid_dropped"] = int(invalid_mask.sum())
    report["total_rows_dropped"] = int(drop_mask.sum())
    report["percent_dropped"] = round(100 * drop_mask.sum() / len(df), 4)

    return clean_df, report
