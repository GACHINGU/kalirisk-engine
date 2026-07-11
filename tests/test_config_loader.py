# tests/test_config_loader.py

from config.loader import load_settings


def test_load_settings_returns_data_contract():
    """
    Check if our settings files loads and has the top-level key we expect.
    """
    settings = load_settings()
    assert "data_contract" in settings


def test_loan_status_is_excluded_from_leakage():
    """
    These is the most important test. If these ever fails, it means
    someone accidentally removed our leakage protection rule.
    """
    settings = load_settings()
    leakage_excluded = settings["data_contract"]["leakage_excluded"]

    assert "loan_status" in leakage_excluded
