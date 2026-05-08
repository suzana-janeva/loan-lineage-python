import numpy as np
import shap
from model import EMPLOYMENT_MAP, _to_feature_array, load_model

FEATURE_LABELS = ['age', 'income', 'loan_amount', 'credit_score', 'employment_status']


def generate_explanation(data: dict, version_name: str = 'rf-v1') -> dict:
    clf = load_model(version_name)

    X = _to_feature_array(data)

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # Handle different SHAP versions (list vs ndarray)
    if isinstance(shap_values, list):
        values = np.array(shap_values[1][0]).flatten()
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:
            values = sv[0, :, 1]
        else:
            values = sv[0]

    # Handle expected_value (scalar vs array)
    expected = explainer.expected_value
    ev_arr   = np.array(expected).flatten()
    base_value = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])

    importance_scores = [
        {
            'feature_name':     feature,
            'importance_score': round(float(v), 6),
        }
        for feature, v in zip(FEATURE_LABELS, values)
    ]
    importance_scores.sort(key=lambda x: abs(x['importance_score']), reverse=True)

    return {
        'version_name': version_name,
        'features':     importance_scores,
        'base_value':   round(base_value, 6),
    }
