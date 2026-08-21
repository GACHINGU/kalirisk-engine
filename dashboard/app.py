# dashboard/app.py

import streamlit as st
import joblib
import requests

model = joblib.load("data/model_artifacts/pd_model_v1.joblib")


def get_options(prefix) -> list:
    """
    These function removes the one-hot encoding DNA from the
    column names, and a gives us text that an underwriter would
    normally write, e.g "verification_status_Not_Verified" to
    "Not Verified".The removal of the underscore in the middle
    of the Not and Verified, needs to be corrected afterwards
    when calling /decide endpoint since the model still requires
    "Not_Verified" as it remembers it that way.
    """
    return [
        name.replace(prefix, "").replace("_", " ")
        for name in model.feature_name_
        if name.startswith(prefix)
    ]


purpose_options = get_options("purpose_")
sub_grade_options = get_options("sub_grade_")
home_ownership_options = get_options("home_ownership_")
verification_status_options = get_options("verification_status_")

st.title("KaliRisk")
st.write()
st.write("The dashboard is alive.")

st.header("New Applicant")


loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
term = st.selectbox("Term", ["36 months", "60 months"])
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
installment = st.number_input("Installments", min_value=0.0, value=300.0)
grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
sub_grade = st.selectbox("Sub Grade", sub_grade_options)
home_ownership = st.selectbox("Home Ownership", home_ownership_options)
annual_inc = st.number_input("Annual Income", min_value=0.0, value=60000.0)
verification_status = st.selectbox("Verification Status", verification_status_options)
purpose = st.selectbox("Purpose", purpose_options)
dti = st.number_input("DTI", min_value=0.0, value=18.5)
fico_range_low = st.number_input("FICO Low", min_value=300, max_value=850, value=700)
fico_range_high = st.number_input("FICO High", min_value=300, max_value=850, value=740)

submitted = st.button("Get Decision")

if submitted:
    payload = {
        "applicants": [
            {
                "loan_amnt": loan_amnt,
                "term": term,
                "int_rate": int_rate,
                "installment": installment,
                "grade": grade,
                "sub_grade": sub_grade,
                "home_ownership": home_ownership,
                "annual_inc": annual_inc,
                "verification_status": verification_status,
                "purpose": purpose,
                "dti": dti,
                "fico_range_low": fico_range_low,
                "fico_range_high": fico_range_high,
            }
        ]
    }

    response = requests.post("http://127.0.0.1:8000/decide", json=payload)
    result = response.json()

    st.subheader("Decision")
    st.metric("Optimal Cutoff", f"{result['best_cutoff']:.2%}")
    st.metric("Expected Profit", f"KES {result['best_profit']:,.2f}")
    st.metric("Approval Rate", f"{result['approval_rate']:.1f}%")
