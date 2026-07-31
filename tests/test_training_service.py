# tests/test_training_serivice.py

import numpy as np
import pandas as pd
import lightgbm as lgb
from services.training_service import train_and_save_pd_model


# a function taking two inputs
# path -> where to save the fake csv
# n_rows -> how many fake loans to generate (defaulting to 50 if not specified)
def make_fake_raw_csv(path, n_rows=50) -> None:
    """
    Builds a small but genuinely trainable fake loan dataset.
    """
    # for reproducibility, making sure the same dataset is generated
    np.random.seed(42)

    dates = pd.date_range("2015-01-01", periods=n_rows, freq="MS")

    df = pd.DataFrame(
        {
            "id": [str(i) for i in range(n_rows)],
            "loan_amnt": np.random.randint(1000, 20000, n_rows),
            "term": np.random.choice(["36 months", "60 months"], n_rows),
            "int_rate": np.random.uniform(5, 25, n_rows),
            "installment": np.random.uniform(100, 800, n_rows),
            "grade": np.random.choice(["A", "B", "C", "D"], n_rows),
            "sub_grade": np.random.choice(["A1", "B2", "C3", "D4"], n_rows),
            "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n_rows),
            "annual_inc": np.random.randint(20000, 150000, n_rows),
            "verification_status": np.random.choice(
                ["Verified", "Not Verified"], n_rows
            ),
            "issue_d": dates.strftime("%b-%Y"),
            "loan_status": np.random.choice(["Fully Paid", "Charged Off"], n_rows),
            "purpose": np.random.choice(
                ["car", "debt_consolidation", "credit_card"], n_rows
            ),
            "dti": np.random.uniform(5, 40, n_rows),
            "fico_range_low": np.random.randint(600, 800, n_rows),
            "fico_range_high": np.random.randint(600, 800, n_rows),
        }
    )
    df.to_csv(path, index=False)


def test_train_and_save_pd_model_creates_new_file_for_the_given_model_dir(
    tmp_path,
) -> None:
    """
    Confirms if the train_and_save_pd_model function, actually creates a new model file, in the
    given model_dir each time its run.
    """
    # create the csv path or address
    # Note these just an address pf where the data should be buts its not yet there
    fake_csv = tmp_path / "scratch_fake_loan_data.csv"

    # make the fake loan data
    # Now the address is no longer empty it has real tangible data in it
    make_fake_raw_csv(str(fake_csv), n_rows=50)

    # create the model path
    model_dir = tmp_path / "test_model"

    # running the training function
    model, auc_score, report = train_and_save_pd_model(str(fake_csv), str(model_dir))

    # assert model_dir was created
    assert model_dir.exists()


def test_train_and_save_pd_model_returns_sane_auc(tmp_path) -> None:
    """
    Confirms auc_score falls within the only mathematically valid
    range for ROC-AUC, without needing to see its exact values in advance.
    """
    # create the fake csv datas address
    fake_csv = tmp_path / "scratch_fake_loan_data.csv"

    # make the actual fake csv file in the created address
    make_fake_raw_csv(str(fake_csv), n_rows=50)

    # make the test models address
    model_dir = tmp_path / "test_model"

    # call the train_and_save_pd_model function that creates the actual model
    model, auc_score, report = train_and_save_pd_model(str(fake_csv), str(model_dir))

    # confirm that auc_score falls within a sane range
    assert 0.0 <= auc_score <= 1.0


def test_train_and_save_pd_model_returns_genuine_usable_model(tmp_path) -> None:
    """
    Confirm that train_and_save_pd_model(), returns a genuine and usable LGBMClassifier.
    """
    fake_csv = tmp_path / "scratch_fake_loan_data.csv"
    make_fake_raw_csv(str(fake_csv), n_rows=50)

    model_dir = tmp_path / "test_model"
    model, auc_score, report = train_and_save_pd_model(str(fake_csv), str(model_dir))

    # check and confirm that the function returns a genuine and usable LGBMClassifier model
    assert isinstance(model, lgb.LGBMClassifier)


def test_train_and_save_pd_model_returns_report_dictionary_with_real_content(
    tmp_path,
) -> None:
    """
    Confirm the returned report is a dictionary with real content. Whether
    certain keys exist.
    """
    fake_csv = tmp_path / "scratch_fake_loan_data.csv"
    make_fake_raw_csv(str(fake_csv), n_rows=50)

    model_dir = tmp_path / "test_model"
    model, auc_score, report = train_and_save_pd_model(str(fake_csv), str(model_dir))

    # assert total rows checked ar 50
    assert report["total_rows_checked"] == 50

    # assert there is a total_rows_dropped key, and confrim if int
    assert "total_rows_dropped" in report
    assert isinstance(report["total_rows_dropped"], int)
