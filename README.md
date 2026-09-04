
# AI Risk Manager — Cost-Aware Payment Fraud Detection

An AI-powered payment fraud detection and risk scoring system designed to help merchants identify potentially fraudulent transactions while balancing fraud detection against false-positive costs.

## 🚀 Project Overview

The system analyzes payment transaction features using a machine learning model and produces:

- **Fraud probability**
- **Risk score (0–100)**
- **Risk level**
- **Recommended action: ALLOW or REVIEW**
- **Feature-based explanation**
- **Cost-aware decision threshold**

The project is designed as a **defense-only fraud detection system**.

## 🏗️ Architecture

```text
Payment Transaction
        ↓
Feature Processing
        ↓
Random Forest Model
        ↓
Fraud Probability
        ↓
Risk Score (0–100)
        ↓
Cost-Based Threshold
        ↓
ALLOW / REVIEW
        ↓
Risk Explanation
````

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset**.

* **284,807 transactions**
* **492 originally labeled fraud transactions**
* **30 input features**
* **Highly imbalanced classification problem**
* `V1–V28` are anonymized PCA-transformed features
* `Amount` represents the transaction amount
* `Time` represents elapsed time

The raw dataset is intentionally **not included in this repository**.

## 🤖 Models Evaluated

Three machine learning models were evaluated:

1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**

Random Forest was selected based on validation **PR-AUC**.

### Validation Results

| Model               | Precision | Recall |     F1 |     PR-AUC |
| ------------------- | --------: | -----: | -----: | ---------: |
| Logistic Regression |     5.30% | 92.50% | 10.03% |     68.75% |
| Random Forest       |    94.12% | 80.00% | 86.49% | **85.34%** |
| XGBoost             |    91.89% | 85.00% | 88.31% |     84.54% |

## 🎯 Cost-Aware Threshold

Because missing a fraudulent transaction can be more expensive than reviewing a legitimate transaction, the system uses a **cost-sensitive threshold**.

Assumptions used:

* **False Positive Cost = 1**
* **False Negative Cost = 10**

The threshold was selected using the **validation set** before evaluating the untouched test set.

**Selected threshold: 0.10**

```text
Fraud probability < 0.10  → ALLOW
Fraud probability ≥ 0.10  → REVIEW
```

## 🧪 Final Held-Out Test Results

The final model was evaluated once on an **untouched temporal test set**.

| Metric              |      Result |
| ------------------- | ----------: |
| Precision           |  **64.84%** |
| Recall              |  **79.73%** |
| F1 Score            |  **71.52%** |
| PR-AUC              |  **82.59%** |
| False Positive Rate | **0.0565%** |

### Confusion Matrix

```text
                 Predicted
              Legit    Fraud

Actual Legit   56640      32
Actual Fraud      15      59
```

The test set contains only **74 fraud transactions**, so the reported metrics should be interpreted with that limitation in mind.

## 💻 Streamlit Dashboard

The project includes an interactive **Streamlit dashboard**.

The dashboard supports:

* **Sample legitimate transaction**
* **Sample fraud transaction**
* **Manual transaction input**
* **Fraud probability**
* **Risk score**
* **Risk classification**
* **Recommended action**
* **Feature importance explanation**
* **Final evaluation metrics**

### Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## 📁 Project Structure

```text
AI-Risk-Manager/
│
├── app.py
├── README.md
├── requirements.txt
│
├── 01_data_exploration.py
├── 02_prepare_data.py
├── 03_train_models.py
├── 04_threshold_analysis.py
├── 05_risk_scoring.py
├── 06_explanation.py
├── 09_evaluation.py
│
├── data/
│   └── Dataset files excluded from GitHub
│
└── model/
    ├── best_fraud_model.joblib
    ├── best_threshold.txt
    ├── feature_importance.csv
    ├── final_test_metrics.csv
    ├── threshold_analysis.csv
    ├── top_10_feature_importance.png
    └── validation_risk_scores.csv
```

## 🔍 Explainability

The Random Forest model's feature importance analysis identifies the most influential anonymized features.

Top features include:

* **V14**
* **V10**
* **V4**
* **V12**
* **V17**

Because `V1–V28` are anonymized PCA components, the project does not assign invented business meanings to individual features.

## 🛡️ Defense-Only Design

This project is strictly designed for **fraud prevention and merchant loss reduction**.

It does not provide methods for bypassing fraud detection, exploiting payment systems, or committing financial abuse.

## ⚠️ Limitations

* The dataset is a **public research dataset** and is not Razorpay transaction data.
* The `V1–V28` features are anonymized.
* The cost values are **project assumptions**, not real merchant loss estimates.
* The model's probability output should be treated as a **model-derived risk score**, rather than a calibrated financial probability.
* Real-world deployment would require monitoring, retraining, drift detection, and business-specific cost modeling.

## 🧰 Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Streamlit**




