import os
import pickle
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR = os.path.dirname(__file__)

FEATURES = ['age', 'income', 'loan_amount', 'credit_score', 'employment_status']

# employment_status integer encoding: 0=employed, 1=self-employed, 2=unemployed, 3=retired
EMPLOYMENT_MAP = {
    'employed':      0,
    'self-employed': 1,
    'unemployed':    2,
    'retired':       3,
}


def _model_path(version_name: str) -> str:
    return os.path.join(MODELS_DIR, f'loan_model_{version_name}.pkl')


def _to_feature_array(data: dict) -> np.ndarray:
    employment = data['employment_status']
    # Accept integer directly or encode from string
    if isinstance(employment, str):
        employment = EMPLOYMENT_MAP.get(employment.lower(), 0)
    return np.array([[
        float(data['age']),
        float(data['income']),
        float(data['loan_amount']),
        float(data['credit_score']),
        float(employment),
    ]])


def train(
    version_name: str = 'rf-v1',
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    training_data: Optional[List[dict]] = None,
) -> dict:
    if training_data:
        df = pd.DataFrame(training_data)
        # Encode string labels if present
        if df['employment_status'].dtype == object:
            df['employment_status'] = df['employment_status'].map(EMPLOYMENT_MAP).fillna(0)
        X = df[FEATURES].values
        y = df['approved'].values
    else:
        rng = np.random.default_rng(42)
        n = 500

        age          = rng.integers(20, 65, n).astype(float)
        income       = rng.uniform(20_000, 150_000, n)
        loan_amount  = rng.uniform(5_000, 100_000, n)
        credit_score = rng.integers(300, 850, n).astype(float)
        employment   = rng.choice(4, n).astype(float)  # 0-3 integers directly

        score = (
            (credit_score - 300) / 550 * 40
            + (income / 150_000) * 30
            + ((1 - loan_amount / 100_000)) * 20
            + (age / 65) * 10
            - (employment == 2) * 15   # penalty for unemployed (2)
        )
        y = (score >= 45).astype(int)
        X = np.column_stack([age, income, loan_amount, credit_score, employment])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    clf.fit(X, y)

    with open(_model_path(version_name), 'wb') as f:
        pickle.dump(clf, f)

    accuracy = float(clf.score(X, y))
    return {
        'status':       'trained',
        'version_name': version_name,
        'n_estimators': n_estimators,
        'max_depth':    max_depth,
        'samples':      len(X),
        'accuracy':     round(accuracy, 4),
    }


def predict(data: dict, version_name: str = 'rf-v1') -> dict:
    path = _model_path(version_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{version_name}' not found. Train it first via POST /train.")

    with open(path, 'rb') as f:
        clf = pickle.load(f)

    X             = _to_feature_array(data)
    prediction    = int(clf.predict(X)[0])
    probabilities = clf.predict_proba(X)[0]
    confidence    = float(probabilities[prediction])

    return {
        'result':       'APPROVED' if prediction == 1 else 'REJECTED',
        'confidence':   round(confidence, 4),
        'version_name': version_name,
    }


def load_model(version_name: str = 'rf-v1'):
    path = _model_path(version_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{version_name}' not found.")
    with open(path, 'rb') as f:
        return pickle.load(f)
