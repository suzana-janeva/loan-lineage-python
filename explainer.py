import numpy as np
import shap
from model import EMPLOYMENT_MAP, _to_feature_array, load_model

FEATURE_LABELS = ['age', 'income', 'loan_amount', 'credit_score', 'employment_status']


def generate_explanation(data: dict, version_name: str = 'rf-v1') -> dict:
    clf = load_model(version_name)

    X = _to_feature_array(data)

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    importance_scores = [
        {
            'feature_name':     feature,
            'importance_score': round(float(score), 6),
        }
        for feature, score in zip(FEATURE_LABELS, values)
    ]
    importance_scores.sort(key=lambda x: abs(x['importance_score']), reverse=True)

    return {
        'version_name': version_name,
        'features':     importance_scores,
        'base_value':   round(float(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, np.ndarray)
            else explainer.expected_value
        ), 6),
    }
