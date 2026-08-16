# app/main.py

from services.decisioning_service import decide_best_cutoff_and_profit
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# load the model once, everytime the server starts, so when a visitor vists the website
# they find it loaded, just sitting there.
# No time wasted reloading the model.
model = joblib.load("data/model_artifacts/pd_model_v1.joblib")

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
