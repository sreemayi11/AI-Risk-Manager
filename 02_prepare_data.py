import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD AND CLEAN DATASET
# ============================================================

print("=" * 60)
print("STEP 1: LOAD DATASET")
print("=" * 60)

df = pd.read_csv("data/creditcard.csv")

print("Original dataset shape:", df.shape)

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

print("Dataset shape after removing duplicates:", df.shape)


# ============================================================
# 2. SORT DATA CHRONOLOGICALLY
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: SORT DATA BY TIME")
print("=" * 60)

df = df.sort_values("Time").reset_index(drop=True)

print("Data sorted by transaction time.")


# ============================================================
# 3. CREATE FINAL HELD-OUT TEST SET
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: CREATE FINAL TEST SET")
print("=" * 60)

# First 80% = development data
# Final 20% = completely untouched test data

cutoff_index = int(len(df) * 0.80)

train_val = df.iloc[:cutoff_index].copy()
test = df.iloc[cutoff_index:].copy()

print("Train + Validation shape:", train_val.shape)
print("Final Test shape:", test.shape)

print("\nTrain + Validation class distribution:")
print(train_val["Class"].value_counts())

print("\nFinal Test class distribution:")
print(test["Class"].value_counts())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: SEPARATE FEATURES AND TARGET")
print("=" * 60)

# Target variable
# 0 = legitimate transaction
# 1 = fraudulent transaction

X = train_val.drop(columns=["Class"])
y = train_val["Class"]

X_test = test.drop(columns=["Class"])
y_test = test["Class"]

print("Development features shape:", X.shape)
print("Development target shape:", y.shape)

print("Test features shape:", X_test.shape)
print("Test target shape:", y_test.shape)


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: TRAIN / VALIDATION SPLIT")
print("=" * 60)

# Split the first 80% into:
# 80% of development data -> training
# 20% of development data -> validation

# Stratify keeps the fraud/legitimate ratio similar
# in both training and validation sets.

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())


# ============================================================
# 6. CHECK CLASS IMBALANCE
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: CLASS IMBALANCE CHECK")
print("=" * 60)

print("\nTraining class percentages:")
print(y_train.value_counts(normalize=True) * 100)

print("\nValidation class percentages:")
print(y_val.value_counts(normalize=True) * 100)

print("\nTest class percentages:")
print(y_test.value_counts(normalize=True) * 100)


# ============================================================
# 7. CHECK FOR MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: CHECK MISSING VALUES")
print("=" * 60)

print("Missing values in training data:", X_train.isnull().sum().sum())
print("Missing values in validation data:", X_val.isnull().sum().sum())
print("Missing values in test data:", X_test.isnull().sum().sum())


# ============================================================
# 8. SAVE PREPARED DATASETS
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: SAVE PREPARED DATASETS")
print("=" * 60)

# Save the datasets separately so the next scripts
# can use them without repeating the preprocessing.

X_train.to_csv("data/X_train.csv", index=False)
X_val.to_csv("data/X_val.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_val.to_csv("data/y_val.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Saved:")
print("  data/X_train.csv")
print("  data/X_val.csv")
print("  data/X_test.csv")
print("  data/y_train.csv")
print("  data/y_val.csv")
print("  data/y_test.csv")


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)

print("\nFinal dataset sizes:")
print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)

print("\nFraud counts:")
print("Training   :", y_train.sum())
print("Validation :", y_val.sum())
print("Test       :", y_test.sum())

print("\nThe final test set has NOT been used for training.")
print("It will remain untouched until final model evaluation.")