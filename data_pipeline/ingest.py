# data_pipeline/ingest.py

import pandas as pd

# Lets call the module loader.py and use the function load_settings()
from config.loader import load_settings


def load_raw_loans(raw_path: str) -> pd.DataFrame:
    """
    Reads the raw loan csv, but only trusts columns listed in our
    data contract (settings.yaml). Fails loudly if an expected
    column is missing, instead of continuing silently.

    Also doesn't care if extra columns are added to the raw dataset.
    """
    # Calling in the python dictionary from load_settings()
    # Assigning a variable name settings to the dictionary
    settings = load_settings()

    # Pulling out approved column lists from the data contract in settings.yaml
    # And assigning them a variable_name "gold_column"
    gold_columns = settings["data_contract"]["raw_columns"]

    # The enforcer, usecols=, pandas will only load the needed columns (gold_columns)
    # even if new extras ar added in the dataest
    # We also enforce an "str" data type on the "id" column
    # dtype = {"id": str} -> specifies that the id column should be read as string
    df = pd.read_csv(raw_path, usecols=gold_columns, dtype={"id": str})

    return df
