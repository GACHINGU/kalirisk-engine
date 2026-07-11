# ====================================================================
# Project: KaliRisk Engine
# Moduel: Data Pipeline - Feature Engineering
# Descrption: Transforms raw string text into clean numerical features
# =====================================================================

import pandas as pd


def engineer_features(input_path, output_path):
    """
    Loads our cleaned dataset, performs feature extraction,
    and saves a vectorized matrix ready for modelling.
    """
    print(f" KaliRisk Features: Loading data from {input_path}...")

    # Load our cleaned dataset, 247mb asset
    # We will also force the ID column to load safely as a string to eliminate any warning completely
    df = pd.read_csv(input_path, dtype={"id": str})

    print(" KaliRisk Features: Transforming text columns to numbers...")
    print()
    # 1. Transform 'term' -> Extract just the digits ('36 months' -> 36)
    # .str.extract(r'(\d+)') looks for the numbers inside the text string
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)

    # 2. Transform 'grade' -> Map alphabetic grades to risk weights (A=1, G=7)
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade_numeric"] = df["grade"].map(grade_map)

    # 2.1 Financial Engineering: Calculate what percentage of monthly income goes towards these installmeent
    # That is the Debt To Income ratio:
    # - First we turn annual income to monthly income by dividing by 12 months
    monthly_income = df["annual_inc"] / 12

    # We use a safe division to avoid any division-by-zero-errors if income is recorded as 0
    df["installment_to_income_ratio"] = df["installment"] / (monthly_income + 1e-5)

    # 3. Create target variable: 'is_default'
    # In risk modeling, 1 means they failed to pay (Defaulted), and 0 means they paid back safely.
    default_statuses = ["Charged Off", "Default", "Late (31-120 days)"]
    df["is_default"] = df["loan_status"].isin(default_statuses).astype(int)

    # Drop the original columns we just transformed, and also, these to save memory space
    columns_to_drop = ["term", "grade", "loan_status"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    print(f" KaliRisk Features: Saving engineered matrix to {output_path}")
    # by default .to_csv, uses mode ="w", meaning it has the capability of overwriting
    df.to_csv(output_path, index=False)
    print()

    print("=" * 65)
    print(" KaliRisk SUCCESS: Feature engineering pipeline complete!")
    print(f" Matrix Shape: {df.shape[0]} rows * {df.shape[1]} columns")
    print(f" Total Defaults Spotted: {df['is_default'].sum()} loans")
    print("=" * 65)


if __name__ == "__main__":
    src_file = "./data/processed/cleaned_accepted_loans.csv"
    dest_file = "./data/processed/features_extracted_loans.csv"

    engineer_features(src_file, dest_file)
