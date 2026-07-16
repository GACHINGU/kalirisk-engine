# domain/risk/train.py

import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


MODEL_PATH = "data/model_artifacts/pd_model_v1.joblib"


def train_pd_model(
    X_train, X_test, y_train, y_test
) -> tuple[lgb.LGBMClassifier, float]:
    """
    Trains a LightGBM pd (Probability of Default) model on chronologically
    split data. Returns the trained model plus an honest ROC-AUC score on
    the held-out genuinely unseen test set.

    This function deliberately does NOT choose an approval threshold -
    that decision belongs to the portifolio optimizer layer, not here
    This model's only job is to output a calibrated probability.
    """

    # model architecture
    model = lgb.LGBMClassifier(
        n_estimators=100,  # Building a 100 small decision trees
        learning_rate=0.05,  # how big should each learning step be, 0.05 is small but more accurate
        max_depth=6,  # Max number of questions each tree can ask
        random_state=42,  # Reproducability, keep the results identical, everytime we run code
        verbose=-1,  # hide LightGBM training messages to keep the work clean
    )

    # train the model
    model.fit(X_train, y_train)

    # return every row but show column 1 only, probability of default
    default_probabilities = model.predict_proba(X_test)[:, 1]

    # ROC-AUC measures how well the model separates risky customers
    # from safe customers
    auc_score = roc_auc_score(y_test, default_probabilities)

    return model, auc_score


def save_model(model: lgb.LGBMClassifier, path: str = MODEL_PATH) -> None:
    """Saves a trained model to disk so it never needs retraining to be reused."""
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH):
    """Loads a previously trained and saved model from disk."""
    return joblib.load(path)
