# tests/test_config_loader.py

from config.loader import load_settings


def test_load_settings_returns_data_contract():
    """
    Check if our settings files loads and has the top-level key we expect.
    """
    settings = (
        load_settings()
    )  # bring in the Python dictionary from loader.py in config
    assert (
        "data_contract" in settings
    )  # check whether the dictionary brought has data_contract


def test_loan_status_is_excluded_from_leakage():
    """
    These is the most important test. If these ever fails, it means
    someone accidentally removed our leakage protection rule.
    """
    settings = load_settings()  # dictionary brought in assigned variable_name settings

    # looking for the values of leakage_excluded
    leakage_excluded = settings["data_contract"]["leakage_excluded"]

    # These confirms whether the loan_status value is one of the values in leakage_excluded
    assert "loan_status" in leakage_excluded
