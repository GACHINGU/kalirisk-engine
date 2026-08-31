# dashboard/app.py

import os
import streamlit as st
import joblib
import requests
import pandas as pd

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


model = joblib.load("data/model_artifacts/pd_model_v1.joblib")


def get_options(prefix) -> list:
    """
    These function removes the one-hot encoding DNA from the
    column names, and a gives us text that an underwriter would
    normally write, e.g "verification_status_Not_Verified" to
    "Not Verified".The removal of the underscore in the middle
    of the Not and Verified so it feels human. These will later
    be passed to /decide which will pass it through prepare_new
    _applicant_data(), which one-hot encodes the word again
    readying it for the model.
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

choice = st.sidebar.radio(
    "Navigate", ["New Applicant", "Batch Decision", "Model Insights"]
)

if choice == "New Applicant":
    st.header("New Applicant")

    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
    term = st.selectbox("Term", ["36 months", "60 months"])
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
    installment = st.number_input("Installments", min_value=0.0, value=300.0)
    grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
    sub_grade = st.selectbox("Sub Grade", sub_grade_options)
    home_ownership = st.selectbox("Home Ownership", home_ownership_options)
    annual_inc = st.number_input("Annual Income", min_value=0.0, value=60000.0)
    verification_status = st.selectbox(
        "Verification Status", verification_status_options
    )
    purpose = st.selectbox("Purpose", purpose_options)
    dti = st.number_input("DTI", min_value=0.0, value=18.5)
    fico_range_low = st.number_input(
        "FICO Low", min_value=300, max_value=850, value=700
    )
    fico_range_high = st.number_input(
        "FICO High", min_value=300, max_value=850, value=740
    )

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

        response = requests.post(f"{API_URL}/decide", json=payload)
        result = response.json()

        st.subheader("Decision")
        st.metric("Optimal Cutoff", f"{result['best_cutoff']:.2%}")
        st.metric("Expected Profit", f"KES {result['best_profit']:,.2f}")
        st.metric("Approval Rate", f"{result['approval_rate']:.1f}%")

elif choice == "Batch Decision":
    st.header("Batch Decision")

    uploaded_file = st.file_uploader("Upload applicant CSV", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(batch_df.head())

        submitted_batch = st.button("Get Batch Decision")

        if submitted_batch:
            payload = {"applicants": batch_df.to_dict(orient="records")}

            response = requests.post(f"{API_URL}/decide", json=payload)
            result = response.json()

            st.subheader("Portfolio Decision")
            st.metric("Optimal Cutoff", f"{result['best_cutoff']:.2%}")
            st.metric("Expected Profit", f"KES {result['best_profit']:,.2f}")
            st.metric("Approval Rate", f"{result['approval_rate']:.1f}%")

        st.subheader("Explore Cutoffs")
        manual_cutoff = st.slider(
            "Try a different cutoff",
            min_value=0.01,
            max_value=0.99,
            value=0.20,
            step=0.01,
        )

        if uploaded_file is not None:
            eval_payload = {
                "applicants": batch_df.to_dict(orient="records"),
                "cutoff": manual_cutoff,
            }
            eval_response = requests.post(f"{API_URL}/evaluate", json=eval_payload)
            eval_result = eval_response.json()

            st.metric("Total Profit", f"KES {eval_result['total_profit']:,.2f}")
            st.metric("Approval Rate", f"{eval_result['approval_rate']:.1f}%")

            el_ratio = eval_result["el_ratio"]
            if el_ratio > 0.08:
                st.error(
                    f"Expected Loss Ratio: {el_ratio:.2%} — exceeds the 8% safety limit!"
                )
            else:
                st.success(f"Expected Loss Ratio: {el_ratio:.2%} — within safety limit")

elif choice == "Model Insights":
    st.header("Model Insights")
    st.write(
        "What the model relies on most, across every applicant it has ever scored."
    )

    response = requests.get(f"{API_URL}/model-insights")
    result = response.json()

    importance_df = pd.DataFrame(
        {
            "feature": result["features"],
            "importance": result["importance"],
        }
    )

    st.bar_chart(importance_df.set_index("feature")["importance"])
