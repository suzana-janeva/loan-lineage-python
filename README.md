# Loan Lineage — Python API

A Flask REST API for loan approval prediction using a Random Forest classifier, with SHAP-based explainability.

## Features

- Train a Random Forest model on loan applicant data
- Predict loan approval with confidence scores
- Explain predictions using SHAP feature importance
- Supports multiple versioned models

## Requirements

```
flask
numpy
pandas
scikit-learn
shap
```

## Running

```bash
python app.py
```

The server starts on `http://0.0.0.0:5000`.

---

## API Endpoints

### `POST /train`

Train a new model version.

**Request body (all optional):**
```json
{
  "version_name": "rf-v1",
  "n_estimators": 100,
  "max_depth": null,
  "training_data": []
}
```

If `training_data` is omitted, synthetic data (500 samples) is used.

Each record in `training_data` must include:

| Field | Type | Description |
|---|---|---|
| `age` | number | Applicant age |
| `income` | number | Annual income |
| `loan_amount` | number | Requested loan amount |
| `credit_score` | number | Credit score (300–850) |
| `employment_status` | string/int | See below |
| `approved` | int | `1` = approved, `0` = rejected |

**Response:**
```json
{
  "status": "trained",
  "version_name": "rf-v1",
  "n_estimators": 100,
  "max_depth": null,
  "samples": 500,
  "accuracy": 0.964
}
```

---

### `POST /predict`

Predict loan approval for an applicant.

**Request body:**
```json
{
  "version_name": "rf-v1",
  "age": 35,
  "income": 75000,
  "loan_amount": 20000,
  "credit_score": 720,
  "employment_status": "employed"
}
```

**Response:**
```json
{
  "result": "APPROVED",
  "confidence": 0.87,
  "version_name": "rf-v1"
}
```

---

### `POST /explain`

Get SHAP feature importance scores for a prediction.

**Request body:** same as `/predict`

**Response:**
```json
{
  "version_name": "rf-v1",
  "base_value": 0.612,
  "features": [
    { "feature_name": "credit_score", "importance_score": 0.152 },
    { "feature_name": "income",       "importance_score": 0.098 },
    { "feature_name": "loan_amount",  "importance_score": -0.071 },
    { "feature_name": "age",          "importance_score": 0.034 },
    { "feature_name": "employment_status", "importance_score": -0.012 }
  ]
}
```

Features are sorted by absolute importance (highest first).

---

## Employment Status Values

| String | Integer |
|---|---|
| `employed` | `0` |
| `self-employed` | `1` |
| `unemployed` | `2` |
| `retired` | `3` |

Both string and integer values are accepted.

## Model Files

Trained models are saved as `loan_model_<version_name>.pkl` in the project directory. These are excluded from version control via `.gitignore`.
