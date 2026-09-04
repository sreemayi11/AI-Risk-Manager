import pandas as pd
import matplotlib.pyplot as plt
import joblib

print("=" * 60)
print("STEP 7: MODEL EXPLAINABILITY")
print("=" * 60)


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained fraud detection model...")

model = joblib.load("model/best_fraud_model.joblib")

print("Model loaded successfully.")
print("Model type:", type(model).__name__)


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv").squeeze()

print("Validation data shape:", X_val.shape)


# ============================================================
# 3. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

feature_importance = pd.DataFrame({
    "Feature": X_val.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 4. DISPLAY TOP FEATURES
# ============================================================

print("\nTop 10 Most Important Features:")
print("-" * 40)

print(feature_importance.head(10).to_string(index=False))


# ============================================================
# 5. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "model/feature_importance.csv",
    index=False
)

print("\nSaved:")
print("model/feature_importance.csv")


# ============================================================
# 6. CREATE TOP 10 FEATURE IMPORTANCE CHART
# ============================================================

top_features = feature_importance.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features Used by Fraud Detection Model")

plt.tight_layout()

plt.savefig(
    "model/top_10_feature_importance.png",
    dpi=300
)

plt.show()

print("\nSaved:")
print("model/top_10_feature_importance.png")


# ============================================================
# 7. FIND A HIGH-RISK TRANSACTION
# ============================================================

print("\nFinding a high-risk validation transaction...")

fraud_probabilities = model.predict_proba(X_val)[:, 1]

results = X_val.copy()

results["Fraud_Probability"] = fraud_probabilities
results["Actual_Class"] = y_val.values

high_risk = results.sort_values(
    by="Fraud_Probability",
    ascending=False
).iloc[0]


# ============================================================
# 8. DISPLAY HIGH-RISK TRANSACTION
# ============================================================

print("\n" + "=" * 60)
print("HIGH-RISK TRANSACTION")
print("=" * 60)

print("\nFraud Probability:",
      round(high_risk["Fraud_Probability"], 4))

print("Risk Score:",
      round(high_risk["Fraud_Probability"] * 100, 2))

print("Actual Class:",
      int(high_risk["Actual_Class"]))

print("\nTransaction Features:")

for feature in X_val.columns:
    print(
        f"{feature}: {high_risk[feature]}"
    )


# ============================================================
# 9. SHOW IMPORTANT FEATURES FOR THIS TRANSACTION
# ============================================================

print("\n" + "=" * 60)
print("IMPORTANT FEATURES FOR MODEL INTERPRETATION")
print("=" * 60)

top_5 = feature_importance.head(5)

for _, row in top_5.iterrows():

    feature = row["Feature"]
    importance = row["Importance"]
    value = high_risk[feature]

    print(
        f"{feature}: value={value}, "
        f"global_importance={importance:.6f}"
    )


# ============================================================
# 10. EXPLAINABILITY NOTE
# ============================================================

print("\n" + "=" * 60)
print("EXPLAINABILITY NOTE")
print("=" * 60)

print("""
The feature importance values represent the overall importance
of each feature to the Random Forest model.

They do NOT mean that a feature directly causes fraud.

The V1-V28 features in this dataset are anonymized PCA-based
features, so their real-world meanings are not available.

Therefore, this project does not assign invented meanings to
these features.
""")


# ============================================================
# 11. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("STEP 7 COMPLETE")
print("=" * 60)

print("\nGenerated files:")
print("  model/feature_importance.csv")
print("  model/top_10_feature_importance.png")

print("\nExplainability analysis completed successfully.")