# tests/test_pipeline.py

from data_pipeline.pipeline import run_data_pipeline


def test_pipeline_runs_end_to_end(tmp_path):
    """
    Confirms ingest -> validate -> engineer_features are wired together
    correctly. Uses a tiny hand-built CSV instead of the real 2.26M-row
    file, since we already trust each individual stage - this test only
    needs to prove the STAGES CONNECT, not re-prove each stage's logic.
    """
    fake_csv = tmp_path / "fake_loans.csv"

    # Lets write a literal csv file with one header row, one single fake loan row
    fake_csv.write_text(
        "id,loan_amnt,term,int_rate,installment,grade,sub_grade,"
        "home_ownership,annual_inc,verification_status,issue_d,loan_status,"
        "purpose,dti,fico_range_low,fico_range_high\n"
        "1,5000,36 months,10.5,150,B,B2,RENT,60000,Verified,Jan-2015,"
        "Fully Paid,debt_consolidation,20,700,750\n"
    )
    features_df, report = run_data_pipeline(str(fake_csv))

    assert len(features_df) == 1
    assert "is_default" in features_df.columns
    assert "loan_status" not in features_df.columns
    assert report["total_rows_checked"] == 1


# tmp_path, these a special tool pytest gives for free
# a temporal folder that exists only fro the duration of
# these one test.
# We use it to create a tiny file on disk since load_raw_loans, requires
# genuine path
