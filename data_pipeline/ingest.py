# data_pipeline/ingest.py

import pandas as pd

# Lets call the module loader.py and use the function load_settings()
from config.loader import load_settings


def load_raw_loans(raw_path: str) -> pd.DataFrame:
    """
    Reads the raw loan csv, but only trusts columns listed in our
    data contract (settings.yaml). Fails loudly is an expected
    columns is missing, instead of continuing silently.
    """
    settings = load_settings()

    # Pulling out approve column lists from the data contract in settings.yaml
    gold_columns = settings["data_contract"]["raw_columns"]

    # The enforcer, usecols=, pandas will only load the needed columns even if new extras ar added in the dataest
    df = pd.read_csv(raw_path, usecols=gold_columns, dtype={"id": str})

    return df
