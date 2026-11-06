"""
Preprocessing stage - Financial Transaction Fraud Detection (PaySim dataset).

Steps:
1. Load the data
2. Inspect it (shape, types, missing values, fraud ratio)
3. Clean it (missing values, duplicates, useless columns)
4. Sanity-check values (amounts)
5. Encode the categorical column `type` (One-Hot)
6. Save the cleaned file

NOT done here on purpose (to avoid data leakage):
- Scaling
- Handling class imbalance (class weights / SMOTE)
These are done AFTER the train/test split, on the training set only.
"""

import pandas as pd

INPUT_FILE = "PS_20174392719_1491204439457_log.csv"
OUTPUT_FILE = "cleaned_fraud_data.csv"

# Account IDs carry no useful signal; isFlaggedFraud is a fixed rule, almost always 0
DROP_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud"]


def load_data(path: str) -> pd.DataFrame:
    """Read the raw CSV file."""
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick overview of the data."""
    print("Shape:", df.shape)
    print("\nColumn types:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicated rows:", df.duplicated().sum())
    print("\nStatistics:")
    print(df.describe())
    if "isFraud" in df.columns:
        print("\nFraud ratio:")
        print(df["isFraud"].value_counts(normalize=True))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove missing values, duplicates and useless columns."""
    before = len(df)

    df = df.dropna()
    print(f"Rows removed (missing values): {before - len(df)}")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Rows removed (duplicates): {before - len(df)}")

    cols = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols)
    print(f"Columns dropped: {cols}")

    return df


def check_values(df: pd.DataFrame) -> None:
    """Sanity checks. Report only: outliers and zero amounts are NOT removed,
    because unusual amounts are often the fraud itself."""
    print("\nRows with amount <= 0:", (df["amount"] <= 0).sum())
    print("Negative balances:",
          (df[["oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest"]] < 0).sum().sum())
    print("Unique transaction types:", df["type"].unique().tolist())


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """One-Hot encode `type` so the model does not treat it as ordered numbers."""
    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], dtype=int)
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save the cleaned data."""
    df.to_csv(path, index=False)
    print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} columns to {path}")


def main() -> None:
    df = load_data(INPUT_FILE)

    print("=== Raw data ===")
    inspect_data(df)

    print("\n=== Cleaning ===")
    df = clean_data(df)
    check_values(df)

    df = encode_data(df)

    print("\n=== Final columns ===")
    print(df.columns.tolist())

    save_data(df, OUTPUT_FILE)
    print("Preprocessing finished successfully.")


if __name__ == "__main__":
    main()
