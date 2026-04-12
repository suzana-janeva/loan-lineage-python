from typing import Optional
from flask import Flask, request, jsonify
from model import train, predict
from explainer import generate_explanation

app = Flask(__name__)

REQUIRED_PREDICT_FIELDS = ['age', 'income', 'loan_amount', 'credit_score', 'employment_status']
VALID_EMPLOYMENT_STRINGS = {'employed', 'self-employed', 'unemployed', 'retired'}
VALID_EMPLOYMENT_INTS    = {0, 1, 2, 3}


def validate_applicant(data: dict) -> Optional[str]:
    for field in REQUIRED_PREDICT_FIELDS:
        if field not in data:
            return f'Missing required field: {field}'
    emp = data['employment_status']
    if isinstance(emp, int):
        if emp not in VALID_EMPLOYMENT_INTS:
            return 'employment_status integer must be 0=employed 1=self-employed 2=unemployed 3=retired'
    else:
        if str(emp) not in VALID_EMPLOYMENT_STRINGS:
            return f"employment_status must be one of: {', '.join(sorted(VALID_EMPLOYMENT_STRINGS))}"
    for field in ['age', 'income', 'loan_amount', 'credit_score']:
        try:
            float(data[field])
        except (ValueError, TypeError):
            return f'{field} must be a number'
    return None


@app.route('/train', methods=['POST'])
def train_route():
    body = request.get_json(silent=True) or {}

    version_name  = body.get('version_name', 'rf-v1')
    n_estimators  = int(body.get('n_estimators', 100))
    max_depth     = body.get('max_depth')
    max_depth     = int(max_depth) if max_depth is not None else None
    training_data = body.get('training_data')

    try:
        result = train(
            version_name=version_name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            training_data=training_data,
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_route():
    data         = request.get_json(silent=True) or {}
    version_name = data.pop('version_name', 'rf-v1')

    error = validate_applicant(data)
    if error:
        return jsonify({'error': error}), 422

    try:
        result = predict(data, version_name=version_name)
        return jsonify(result), 200
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain_route():
    data         = request.get_json(silent=True) or {}
    version_name = data.pop('version_name', 'rf-v1')

    error = validate_applicant(data)
    if error:
        return jsonify({'error': error}), 422

    try:
        result = generate_explanation(data, version_name=version_name)
        return jsonify(result), 200
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
