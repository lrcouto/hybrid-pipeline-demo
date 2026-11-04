import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def split_data(claim_history: pd.DataFrame, params: dict) -> tuple:
    """Split past claims into train and test sets."""
    features = params["features"]["categorical"] + params["features"]["numeric"]
    return train_test_split(
        claim_history[features],
        claim_history[params["target"]],
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=claim_history[params["target"]],
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> Pipeline:
    """Train the claim triage classifier."""
    preprocess = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), params["features"]["categorical"])],
        remainder="passthrough",
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", GradientBoostingClassifier(random_state=params["random_state"])),
        ]
    )
    return model.fit(X_train, y_train)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score the model on held-out claims."""
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, model.predict(X_test)), 3),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 3),
        "test_rows": len(y_test),
    }
