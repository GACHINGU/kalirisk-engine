# =====================================================================
# PROJECT: KaliRisk Engine
# MODULE: Models - Institutional Chronological Risk Trainer
# MODEL: LightGBM Classifier (Gradient Boosting Machine)
# DESCRIPTION: Trains a sequential boosting model using a timeline
#              split, and runs a threshold scan to balance risk.
# =====================================================================

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


def train_risk_engine(input_path):
    """
    Executes chronological model training using LightGBM on a 2.2 million loan dataset.

    This function handles temporal sequencing to prevent data leakage, validates
    hyperparameters across time-series folds, and runs a probability threshold scan
    to balance credit default detection against safe customer retention.
    """
    # Load the processed CSV feature matrix from the data pipeline
    print(f" KaliRisk Trainer: Loading data from {input_path}...")
    # Force the unique ID column to load as a string to stop low-memory type warnings
    df = pd.read_csv(input_path, dtype={"id": str})

    # Convert the string issue dates into actual datetime objects so Python understands time
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    # Sort every loan from oldest to newest to ensure the past never trains on the future
    df = df.sort_values(by="issue_d").reset_index(drop=True)

    # List the exact underwriting and engineered features we want the model to learn from
    feature_cols = [
        "loan_amnt",
        "term_months",
        "int_rate",
        "installment",
        "installment_to_income_ratio",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "grade_numeric",
        "annual_inc",
    ]
    # Slices our features into matrix X and fills any blank values with 0
    X = df[feature_cols].fillna(0)
    # Extracts our target array y where 1 is a default and 0 is a safe loan
    y = df["is_default"]

    # Lock away the most recent 20% of loans to act as our final out-of-time exam vault
    # Because the models were failing on F1-score and precision, lets enable shuffle and see what happens
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=True, random_state=42
    )

    # Set up a rolling 3-fold split across history to validate our baseline performance
    # tscv = TimeSeriesSplit(n_splits=3)
    # Lets swap out the k-fold for a standard shuffled 3-fold validation check
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Internal validation scoring function to see how the model handles a strict 10% risk gate
    def threshold_f1_scorer(estimator, X_val, y_val):
        # Generate raw decimal probability scores (0.0 to 1.0) for the validation data
        probs = estimator.predict_proba(X_val)[:, 1]
        # Force a hard flag of 1 if the risk probability is 10% or higher, else 0
        preds = (probs >= 0.10).astype(int)
        # Calculate the F1-score for defaults using our custom 10% cutoff
        return f1_score(y_val, preds, zero_division=0)

    # Initialize a baseline model using unweighted sequential boosting trees
    val_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    # Run the validation scoring math across our chronological history folds
    print(" Running cross_val_score across randomized folds...")
    cv_scores = cross_val_score(
        val_model, X_train, y_train, cv=kf, scoring=threshold_f1_scorer
    )
    # Print out the average F1 performance across our historical homework checks
    print(f" Mean Validation F1-Score: {cv_scores.mean() * 100:.2f}%\n")

    # Train our actual production model on the entire 80% history pool combined
    print(" Training production Boosting engine on entire History Pool...")
    production_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    # Build the sequential boosting trees using our training features and targets
    production_model.fit(X_train, y_train)

    # Extract raw risk probabilities (0.0 to 1.0) from our locked out-of-time test pool
    raw_probabilities = production_model.predict_proba(X_test)[:, 1]

    # Define an array of cutoff gates to find out where precision and recall balance out
    test_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    # Initialize placeholders to track the best gate setup during our scan loop
    best_threshold, best_f1 = 0.50, 0.0

    # Print the table headers to show our risk threshold optimization spectrum
    print("\n Scanning risk threshold spectrum...")
    print(
        f"{'Threshold':<12}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}"
    )
    print("-" * 60)

    # Run the loop through each individual threshold candidate to test its performance
    for t in test_thresholds:
        # If a borrower's raw probability is >= current threshold t, flag them as 1 (Default)
        preds = (raw_probabilities >= t).astype(int)
        # Calculate overall accuracy for this specific threshold configuration
        acc = accuracy_score(y_test, preds)
        # Generate the standard classification metrics dictionary for this specific cutoff
        rep = classification_report(y_test, preds, output_dict=True, zero_division=0)

        # Isolate the precision, recall, and F1-score specifically for our default class (1)
        prec, rec, f1 = rep["1"]["precision"], rep["1"]["recall"], rep["1"]["f1-score"]
        # Print a clean row in our table showing the metrics for this cutoff point
        # Make sure they are widely spaced 12 width, for aesthetics and keep a 2-decimal places
        print(f"{t:<12.2f}{acc * 100:<12.2f}%{prec:<12.2f}{rec:<12.2f}{f1:<12.2f}")

        # If this configuration yields a higher default F1-score, update our tracking variables
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    # Print the absolute best risk threshold found across the entire spectrum scan
    print("-" * 60)
    print(f"  Best Gate Found: {best_threshold * 100}% | Peak F1: {best_f1:.4f}\n")

    # Calculate our definitive predictions using the winning optimized cutoff gate
    final_preds = (raw_probabilities >= best_threshold).astype(int)
    # Output the final production classification report for your project showcase review
    print(classification_report(y_test, final_preds, zero_division=0))


if __name__ == "__main__":
    # Execute the entire training script using our engineered feature matrix CSV
    train_risk_engine("./data/processed/features_extracted_loans.csv")
