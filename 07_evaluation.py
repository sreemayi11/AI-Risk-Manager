import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# STEP 9: FINAL HELD-OUT TEST EVALUATION
# ============================================================

print("=" * 70)
print("STEP 9: FINAL HELD-OUT TEST EVALUATION")
print("=" * 70)


# ============================================================
# 1. LOAD FROZEN MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(
    "model/best_fraud_model.joblib"
)

print("Model:", type(model).__name__)


# ============================================================
# 2. LOAD FROZEN THRESHOLD
# ============================================================

with open("model/best_threshold.txt", "r") as file:
    threshold = float(file.read().strip())

print("Decision threshold:", threshold)


# ============================================================
# 3. LOAD FINAL TEST SET
# ============================================================

print("\nLoading untouched final test set...")

X_test = pd.read_csv(
    "data/X_test.csv"
)

y_test = pd.read_csv(
    "data/y_test.csv"
).squeeze()

print("Test features shape:", X_test.shape)
print("Test target shape:", y_test.shape)

print("\nTest class distribution:")
print(y_test.value_counts())


# ============================================================
# 4. GENERATE FRAUD PROBABILITIES
# ============================================================

print("\nGenerating predictions...")

fraud_probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 5. APPLY FROZEN THRESHOLD
# ============================================================

y_pred = (
    fraud_probabilities >= threshold
).astype(int)


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

pr_auc = average_precision_score(
    y_test,
    fraud_probabilities
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()


# ============================================================
# 8. FALSE POSITIVE RATE
# ============================================================

false_positive_rate = fp / (fp + tn)


# ============================================================
# 9. COST ANALYSIS
# ============================================================

# Same project-level assumptions used during threshold analysis.
FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 10

estimated_cost = (
    fp * FALSE_POSITIVE_COST
    + fn * FALSE_NEGATIVE_COST
)


# ============================================================
# 10. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nPrecision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"PR-AUC    : {pr_auc:.4f} ({pr_auc * 100:.2f}%)")

print("\nConfusion Matrix:")
print("-----------------")

print(f"True Negatives  (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives  (TP): {tp}")

print(
    f"\nFalse Positive Rate: "
    f"{false_positive_rate:.6f} "
    f"({false_positive_rate * 100:.4f}%)"
)

print("\nCost Analysis:")
print("-----------------")
print("False Positive Cost:", FALSE_POSITIVE_COST)
print("False Negative Cost:", FALSE_NEGATIVE_COST)
print("Estimated Cost:", estimated_cost)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 12. SAVE FINAL METRICS
# ============================================================

final_metrics = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1 Score",
        "PR-AUC",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
        "False Positive Rate",
        "Estimated Cost",
        "Decision Threshold"
    ],
    "Value": [
        precision,
        recall,
        f1,
        pr_auc,
        tn,
        fp,
        fn,
        tp,
        false_positive_rate,
        estimated_cost,
        threshold
    ]
})

final_metrics.to_csv(
    "model/final_test_metrics.csv",
    index=False
)


# ============================================================
# 13. COMPLETION
# ============================================================

print("\nSaved:")
print("model/final_test_metrics.csv")

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION COMPLETE")
print("=" * 70)

print("""
IMPORTANT:
The final test set was kept separate from model training
and model/threshold selection.

These results are the final held-out evaluation results.
Do not modify the model based on these test results.
""")
