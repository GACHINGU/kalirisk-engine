# app/main.py

from services.decisioning_service import (
    decide_best_cutoff_and_profit,
    evaluate_applicants_at_cutoff,
    prepare_and_score_applicants,
)
from services.training_service import train_and_save_pd_model
from domain.risk.explain import get_feature_importance
from domain.risk.explain import explain_applicant
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# load the model once, everytime the server starts, so when a visitor vists the website
# they find it loaded, just sitting there.
# No time wasted reloading the model.
model = joblib.load("data/model_artifacts/pd_model_v1.joblib")

# a hard-coded path, protects me from a path traversal risk or attack
RAW_DATA_PATH = "data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"

app = FastAPI()


# a standard check just to see if the server is alive
# a GET request
@app.get("/")
def read_root():
    return {"message": "KaliRisk is alive"}


# make sure  applicant data meets the specific requirements
# BaseModel the trained mercenary at enforcing the requirements
class Applicant(BaseModel):
    loan_amnt: float
    term: str
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    dti: float
    fico_range_low: int
    fico_range_high: int


# make sure the data of applicants meets the list and Applicant requirements
# ofcourse BaseModel enforces the requirements
class ApplicantBatch(BaseModel):
    applicants: list[Applicant]


@app.post("/decide")
def decide(batch: ApplicantBatch):
    """
    These functions task is to take the data gotten at the POST request,
    first checks if it meets ApplicantBatch requirements, fails loud if
    requirements not met, if met the data is given the "batch" name and
    then trasformed from a pydantic object to boring dictionary.

    After transformation, its then passed through the decisioning service
    function where its processed and analyzed to give the best cutoff, best
    profit and a full report
    """
    applicants_as_dicts = [applicant.model_dump() for applicant in batch.applicants]

    df = pd.DataFrame(applicants_as_dicts)

    best_cutoff, best_profit, approval_rate, full_report = (
        decide_best_cutoff_and_profit(df=df, model=model)
    )

    return {
        "best_cutoff": best_cutoff,
        "best_profit": best_profit,
        "approval_rate": approval_rate,
        "report": full_report,
    }


@app.post("/train")
def train():
    """
    Retrains the PD model on the fixed, known dataset path - the caller
    has no ability to specify which files gets read, closing off path
    traversal risk entirely.
    """
    model, auc_score, report = train_and_save_pd_model(RAW_DATA_PATH)

    return {"auc_score": auc_score, "report": report}


# make sure applicant data and cutoff meets specified requirements
# BaseModel enforces them
class CutoffRequest(BaseModel):
    applicants: list[Applicant]
    cutoff: float


@app.post("/evaluate")
def evaluate(request: CutoffRequest):
    """
    Evaluates a batch of applicants at ONE specific, caller-chosen
    cutoff - powers the interactive slider. Unlike /decide, these
    never searches for an optimal cutoff; it honestly reports what
    happens at exactly the cutoff given, including breaching max_el_ratio.
    """
    applicants_as_dicts = [applicant.model_dump() for applicant in request.applicants]
    df = pd.DataFrame(applicants_as_dicts)

    total_profit, approval_rate, el_ratio, full_report = evaluate_applicants_at_cutoff(
        df=df, model=model, cutoff=request.cutoff
    )

    return {
        "cutoff": request.cutoff,
        "total_profit": total_profit,
        "approval_rate": approval_rate,
        "el_ratio": el_ratio,
        "report": full_report,
    }


# designing a GET request for model's feature importance
# will display global explainability
@app.get("/model-insights")
def model_insights(top_n: int = 10):
    """
    Returns the model's global feature importance - which features it
    relies on most, across every applicant it has ever scored. Read-only,
    needs no input, since it's a property of the trained model itself.
    top_n is an optional query parameter, defaulting to 10.
    """
    importance_df = get_feature_importance(model=model, top_n=top_n)

    return {
        "features": importance_df["feature"].tolist(),
        "importance": importance_df["importance"].tolist(),
    }


@app.post("/explain")
def explain(applicant: Applicant, top_n: int = 10):
    """
    Explains ONE specific applicant's prediction using SHAP - answers
    "why did I get this score," unlike /model-insights which describes
    the model's behaviour globally, across everyone it has ever scored.
    """
    df = pd.DataFrame([applicant.model_dump()])

    pd_default, expected_loss, profit_per_loan, loan_amnt, report, X_new = (
        prepare_and_score_applicants(df, model)
    )

    explanation_df = explain_applicant(model, X_new, top_n=top_n)

    return {
        "predicted_pd": pd_default[0],
        "features": explanation_df["feature"].tolist(),
        "shap_values": explanation_df["shap_value"].tolist(),
    }
