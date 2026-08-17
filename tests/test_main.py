# tests/test_main.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def make_valid_applicant() -> dict:
    """One realistic, fully valid applicant - matches every field Applicant requires."""
    return {
        "loan_amnt": 10000,
        "term": "36 months",
        "int_rate": 12.5,
        "installment": 335.0,
        "grade": "B",
        "sub_grade": "B2",
        "home_ownership": "RENT",
        "annual_inc": 60000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 18.5,
        "fico_range_low": 700,
        "fico_range_high": 740,
    }


def test_decide_with_valid_applicant_returns_200() -> None:
    """Confirms a properly shaped applicant batch is accepted and returns a real decision."""
    payload = {"applicants": [make_valid_applicant()]}

    response = client.post("/decide", json=payload)

    assert response.status_code == 200

    body = response.json()
    assert "best_cutoff" in body
    assert "best_profit" in body
    assert "approval_rate" in body
    assert "report" in body


def test_decide_with_invalid_data_returns_422() -> None:
    """
    Confirms pydantic rejects malformed applicant data BEFORE decide()
    ever runs - loan_amnt as text instead of a number should fail validation.
    """
    broken_applicant = make_valid_applicant()
    broken_applicant["loan_amnt"] = "ten thousand"

    payload = {"applicants": [broken_applicant]}

    response = client.post("/decide", json=payload)

    assert response.status_code == 422
