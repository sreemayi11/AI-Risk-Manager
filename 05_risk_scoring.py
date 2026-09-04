import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. LOAD MODEL AND THRESHOLD
# ============================================================

print("=" * 70)
print("STEP 1: LOAD MODEL AND THRESHOLD")
print("=" * 70)

model = joblib.load("model/best_fraud_model.joblib")

with open("model/best_threshold.txt", "r") as file:
    fraud_threshold = float(file.read().strip())

print("Model loaded successfully.")
print("Fraud threshold:", fraud_threshold)


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: LOAD VALIDATION DATA")
print("=" * 70)

X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv").squeeze()

print("Validation data:", X_val.shape)


# ============================================================
# 3. GENERATE FRAUD PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: GENERATE FRAUD PROBABILITIES")
print("=" * 70)

fraud_probability = model.predict_proba(X_val)[:, 1]

print("Fraud probabilities generated.")


# ============================================================
# 4. CONVERT PROBABILITY INTO RISK SCORE
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: CREATE RISK SCORE")
print("=" * 70)

# Convert probability from 0-1 into a 0-100 risk score.

risk_score = fraud_probability * 100


# ============================================================
# 5. ASSIGN RISK LEVEL
# ============================================================

def get_risk_level(score):

    if score < 30:
        return "LOW"

    elif score < 70:
        return "MEDIUM"

    else:
        return "HIGH"


risk_level = np.array([
    get_risk_level(score)
    for score in risk_score
])


# ============================================================
# 6. ASSIGN RECOMMENDED ACTION
# ============================================================

def get_action(probability):

    # Below the fraud threshold:
    # allow the transaction.

    if probability < fraud_threshold:
        return "ALLOW"

    # Transactions above the threshold
    # should be investigated rather than
    # automatically rejected.

    else:
        return "REVIEW"


recommended_action = np.array([
    get_action(probability)
    for probability in fraud_probability
])


# ============================================================
# 7. CREATE RISK RESULTS TABLE
# ============================================================

results = pd.DataFrame({
    "Actual_Class": y_val.values,
    "Fraud_Probability": fraud_probability,
    "Risk_Score": risk_score,
    "Risk_Level": risk_level,
    "Recommended_Action": recommended_action
})


# ============================================================
# 8. DISPLAY SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: SAMPLE RISK RESULTS")
print("=" * 70)

print(
    results.head(20).to_string(index=False)
)


# ============================================================
# 9. RISK LEVEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: RISK LEVEL DISTRIBUTION")
print("=" * 70)

print(
    results["Risk_Level"].value_counts()
)


# ============================================================
# 10. ACTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: RECOMMENDED ACTION DISTRIBUTION")
print("=" * 70)

print(
    results["Recommended_Action"].value_counts()
)


# ============================================================
# 11. FRAUD DETECTION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: FRAUD DETECTION SUMMARY")
print("=" * 70)

actual_fraud = results["Actual_Class"] == 1
predicted_fraud = results["Recommended_Action"] == "REVIEW"

true_positives = (
    actual_fraud & predicted_fraud
).sum()

false_positives = (
    (~actual_fraud) & predicted_fraud
).sum()

false_negatives = (
    actual_fraud & (~predicted_fraud)
).sum()

true_negatives = (
    (~actual_fraud) & (~predicted_fraud)
).sum()


print("True Positives :", true_positives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)
print("True Negatives :", true_negatives)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results.to_csv(
    "model/validation_risk_scores.csv",
    index=False
)

print("\nRisk results saved to:")
print("model/validation_risk_scores.csv")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK SCORING COMPLETE")
print("=" * 70)

print("""
Risk scoring pipeline:

Transaction
     ↓
Fraud Probability
     ↓
Risk Score (0-100)
     ↓
Risk Level
     ↓
Recommended Action

LOW / MEDIUM / HIGH

ALLOW / REVIEW

The final test set was NOT used.
""")