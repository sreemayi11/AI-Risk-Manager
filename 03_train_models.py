import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

import joblib
import os


# ============================================================
# 1. LOAD PREPARED DATA
# ============================================================

print("=" * 70)
print("STEP 1: LOAD PREPARED DATA")
print("=" * 70)

X_train = pd.read_csv("data/X_train.csv")
X_val = pd.read_csv("data/X_val.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze()
y_val = pd.read_csv("data/y_val.csv").squeeze()

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())


# ============================================================
# 2. CALCULATE CLASS IMBALANCE WEIGHT
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: CALCULATE CLASS IMBALANCE WEIGHT")
print("=" * 70)

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("Legitimate transactions:", negative_count)
print("Fraud transactions:", positive_count)
print("scale_pos_weight:", scale_pos_weight)


# ============================================================
# 3. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs("model", exist_ok=True)


# ============================================================
# 4. DEFINE MODELS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: DEFINE MODELS")
print("=" * 70)

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

print("Models created:")
for model_name in models:
    print("-", model_name)


# ============================================================
# 5. TRAIN AND EVALUATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: TRAIN AND EVALUATE MODELS")
print("=" * 70)

results = {}

for name, model in models.items():

    print("\n" + "-" * 70)
    print("Training:", name)
    print("-" * 70)

    # Train only on training data
    model.fit(X_train, y_train)

    # Get fraud probabilities
    y_val_probability = model.predict_proba(X_val)[:, 1]

    # Default threshold = 0.50
    y_val_prediction = (y_val_probability >= 0.50).astype(int)

    # Metrics
    precision = precision_score(
        y_val,
        y_val_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_prediction,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y_val,
        y_val_probability
    )

    cm = confusion_matrix(
        y_val,
        y_val_prediction
    )

    # Store results
    results[name] = {
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }

    # Print results
    print("\nValidation Results")
    print("------------------")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            y_val_prediction,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: MODEL COMPARISON")
print("=" * 70)

comparison = []

for name, result in results.items():

    comparison.append({
        "Model": name,
        "Precision": result["precision"],
        "Recall": result["recall"],
        "F1": result["f1"],
        "PR-AUC": result["pr_auc"]
    })

comparison_df = pd.DataFrame(comparison)

comparison_df = comparison_df.sort_values(
    by="PR-AUC",
    ascending=False
)

print("\n")
print(comparison_df.to_string(index=False))


# ============================================================
# 7. SELECT BEST MODEL
# ============================================================

best_model_name = comparison_df.iloc[0]["Model"]

best_model = results[best_model_name]["model"]

print("\n" + "=" * 70)
print("STEP 6: BEST MODEL")
print("=" * 70)

print("Best model based on validation PR-AUC:")
print(best_model_name)

print(
    f"Validation PR-AUC: "
    f"{results[best_model_name]['pr_auc']:.4f}"
)


# ============================================================
# 8. SAVE BEST MODEL
# ============================================================

model_path = "model/best_fraud_model.joblib"

joblib.dump(
    best_model,
    model_path
)

print("\nBest model saved to:")
print(model_path)


# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("- Models were trained only on X_train and y_train.")
print("- Validation data was used only for evaluation.")
print("- The final test set was NOT loaded or used.")
print("- Model selection was based on validation PR-AUC.")