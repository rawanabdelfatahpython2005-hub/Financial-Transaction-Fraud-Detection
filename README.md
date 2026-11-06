# Financial Transaction Fraud Detection - Preprocessing

## Dataset
- PaySim dataset (Kaggle), 6,362,620 rows.
- Cleaned data (Google Drive): https://drive.google.com/file/d/12o9lATWzv7lQLxc5ffCa9x5Uoj9gIyq7/view?usp=sharing
- Data is not stored here because of its size.

## Steps
1. Load the raw CSV and inspect it.
2. Missing values: none found. Duplicates: none found.
3. Dropped columns: nameOrig, nameDest, isFlaggedFraud.
4. Checked amounts: 16 rows with amount = 0 (kept, they are fraud cases).
5. One-Hot encoded the `type` column.
6. Saved cleaned_fraud_data.csv (6,362,620 rows x 12 columns).

## Decisions
- Outliers are kept: unusual amounts are often the fraud itself.
- Fraud is rare (0.13%), so use Precision, Recall, F1 and PR-AUC, not accuracy.
- Scaling and imbalance handling are done after the train/test split, on the training set only, to avoid data leakage.

## Run
pip install -r requirements.txt
python fraud.py
