import pandas as pd

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/creditcard.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()


# ============================================================
# 2. CLASS DISTRIBUTION
# ============================================================

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)


# ============================================================
# 3. TRANSACTION AMOUNT ANALYSIS
# ============================================================

print("\nAmount Statistics:")
print(df.groupby("Class")["Amount"].describe())

print("\nAverage Amount by Class:")
print(df.groupby("Class")["Amount"].mean())

print("\nMinimum Amount by Class:")
print(df.groupby("Class")["Amount"].min())

print("\nMaximum Amount by Class:")
print(df.groupby("Class")["Amount"].max())


# ============================================================
# 4. CHECK DUPLICATES
# ============================================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDuplicate rows by class:")
duplicates = df[df.duplicated(keep=False)]
print(duplicates["Class"].value_counts())

print("\nFraud among duplicated rows:")
print(duplicates[duplicates["Class"] == 1].shape[0])

print("\nDuplicate fraud rows:")
print(
    df[
        (df.duplicated(keep=False)) &
        (df["Class"] == 1)
    ]
)


# ============================================================
# 5. REMOVE EXACT DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nAfter removing exact duplicates:")
print("Dataset Shape:", df.shape)

print("\nDuplicates remaining:")
print(df.duplicated().sum())

print("\nClass Distribution after removing duplicates:")
print(df["Class"].value_counts())

print("\nFraud Percentage after removing duplicates:")
print(df["Class"].value_counts(normalize=True) * 100)


# ============================================================
# 6. TIME ANALYSIS
# ============================================================

print("\nTime Statistics:")
print(df["Time"].describe())

print("\nFraud transactions by time quartile:")
print(
    pd.qcut(
        df["Time"],
        q=4,
        duplicates="drop"
    ).groupby(df["Class"]).value_counts()
)


# ============================================================
# 7. TIME-BASED TRAIN/TEST SPLIT CHECK
# ============================================================

# Sort transactions chronologically
df = df.sort_values("Time").reset_index(drop=True)

# Use the first 80% for model development
# Use the final 20% as the untouched test set
cutoff_index = int(len(df) * 0.80)

cutoff_time = df.iloc[cutoff_index]["Time"]

print("\n80% Time-Based Split")
print("Cutoff index:", cutoff_index)
print("Cutoff time:", cutoff_time)

print("\nTraining + Validation portion:")
print(df.iloc[:cutoff_index]["Class"].value_counts())

print("\nFinal Test portion:")
print(df.iloc[cutoff_index:]["Class"].value_counts())


# ============================================================
# 8. CREATE TRAIN+VALIDATION AND TEST DATASETS
# ============================================================

train_val = df.iloc[:cutoff_index].copy()
test = df.iloc[cutoff_index:].copy()

print("\nFinal Dataset Split")
print("-------------------")

print("\nTrain + Validation:")
print("Shape:", train_val.shape)
print(train_val["Class"].value_counts())

print("\nFinal Test:")
print("Shape:", test.shape)
print(test["Class"].value_counts())