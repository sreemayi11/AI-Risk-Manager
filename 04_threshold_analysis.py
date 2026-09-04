import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("STEP 1: LOAD VALIDATION DATA")
print("=" * 70)

X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv").squeeze()

print("Validation features:", X_val.shape)
print("Validation target:", y_val.shape)

print("\nFraud cases in validation set:", y_val.sum())


# ============================================================
# 2. LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: LOAD RANDOM FOREST MODEL")
print("=" * 70)

model = joblib.load("model/best_fraud_model.joblib")

print("Model loaded successfully.")


# ============================================================
# 3. GET FRAUD PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: GENERATE FRAUD PROBABILITIES")
print("=" * 70)

# Probability that each transaction is fraudulent
fraud_probability = model.predict_proba(X_val)[:, 1]

print("Fraud probabilities generated.")

print("\nProbability statistics:")
print(pd.Series(fraud_probability).describe())


# ============================================================
# 4. CALCULATE PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_val,
    fraud_probability
)

print("\nValidation PR-AUC:", round(pr_auc, 4))


# ============================================================
# 5. TEST DIFFERENT THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

threshold_results = []

for threshold in thresholds:

    # Convert probabilities into fraud predictions
    predictions = (
        fraud_probability >= threshold
    ).astype(int)

    # Calculate metrics
    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
        "True Negatives": tn
    })


threshold_df = pd.DataFrame(threshold_results)


print("\nThreshold comparison:")
print(
    threshold_df.to_string(
        index=False,
        formatters={
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)


# ============================================================
# 6. COST MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: COST-AWARE ANALYSIS")
print("=" * 70)

print("""
We need to represent the business cost of two mistakes:

False Positive:
A legitimate transaction is flagged as suspicious.

False Negative:
A fraudulent transaction is allowed through.

These costs are project assumptions, NOT Razorpay's actual internal costs.
""")


# ------------------------------------------------------------
# Project cost assumptions
# ------------------------------------------------------------

# Cost assigned when a legitimate transaction
# is incorrectly flagged.
FALSE_POSITIVE_COST = 1

# Cost assigned when a fraudulent transaction
# is missed.
FALSE_NEGATIVE_COST = 10


print("False Positive Cost :", FALSE_POSITIVE_COST)
print("False Negative Cost :", FALSE_NEGATIVE_COST)


# Calculate total estimated cost
threshold_df["Estimated Cost"] = (
    threshold_df["False Positives"] * FALSE_POSITIVE_COST
    +
    threshold_df["False Negatives"] * FALSE_NEGATIVE_COST
)


print("\nCost-aware threshold comparison:")

print(
    threshold_df[
        [
            "Threshold",
            "Precision",
            "Recall",
            "F1",
            "False Positives",
            "False Negatives",
            "Estimated Cost"
        ]
    ].to_string(
        index=False,
        formatters={
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)


# ============================================================
# 7. FIND LOWEST-COST THRESHOLD
# ============================================================

best_cost_row = threshold_df.loc[
    threshold_df["Estimated Cost"].idxmin()
]

best_threshold = best_cost_row["Threshold"]
lowest_cost = best_cost_row["Estimated Cost"]


print("\n" + "=" * 70)
print("STEP 6: BEST COST-AWARE THRESHOLD")
print("=" * 70)

print("Best threshold:", best_threshold)
print("Lowest estimated cost:", lowest_cost)

print("\nMetrics at selected threshold:")

print(
    f"Precision       : {best_cost_row['Precision']:.4f}"
)

print(
    f"Recall          : {best_cost_row['Recall']:.4f}"
)

print(
    f"F1 Score        : {best_cost_row['F1']:.4f}"
)

print(
    f"False Positives  : {int(best_cost_row['False Positives'])}"
)

print(
    f"False Negatives  : {int(best_cost_row['False Negatives'])}"
)

print(
    f"True Positives   : {int(best_cost_row['True Positives'])}"
)


# ============================================================
# 8. SAVE THRESHOLD RESULTS
# ============================================================

threshold_df.to_csv(
    "model/threshold_analysis.csv",
    index=False
)

print("\nThreshold analysis saved to:")
print("model/threshold_analysis.csv")


# ============================================================
# 9. SAVE SELECTED THRESHOLD
# ============================================================

with open("model/best_threshold.txt", "w") as file:
    file.write(str(best_threshold))

print("Selected threshold saved to:")
print("model/best_threshold.txt")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print("""
Important:
- The final test set was NOT used.
- Threshold selection was performed only on validation data.
- The selected threshold minimizes the assumed business cost.
- The cost values are project assumptions and can be changed later.
""")