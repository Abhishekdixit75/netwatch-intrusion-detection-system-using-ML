"""
ml_pipeline/preprocess_unsw.py

Preprocesses UNSW-NB15 parquet files and saves:
  - data/processed/unsw_train.parquet
  - data/processed/unsw_test.parquet
  - ml_pipeline/models/label_encoder_classes.json
  - ml_pipeline/models/scaler_params.json
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_TRAIN  = "data/raw/UNSW_NB15/UNSW_NB15_training-set.parquet"
RAW_TEST   = "data/raw/UNSW_NB15/UNSW_NB15_testing-set.parquet"
OUT_DIR    = "data/processed"
MODEL_DIR  = "ml_pipeline/models"

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(RAW_TRAIN)
test  = pd.read_parquet(RAW_TEST)
print(f"  Train: {train.shape}  |  Test: {test.shape}")

# ── 2. Drop duplicates (train only) ──────────────────────────────────────────
before = len(train)
train = train.drop_duplicates()
print(f"Dropped {before - len(train):,} duplicates → {len(train):,} rows remain")

# ── 3. Strip whitespace from attack_cat (common issue in this dataset) ────────
train["attack_cat"] = train["attack_cat"].str.strip()
test["attack_cat"]  = test["attack_cat"].str.strip()

# ── 4. Verify no missing values ───────────────────────────────────────────────
assert train.isnull().sum().sum() == 0, "Missing values found in train!"
assert test.isnull().sum().sum()  == 0, "Missing values found in test!"
print("No missing values — confirmed.")

# ── 5. Identify feature columns ───────────────────────────────────────────────
# Drop target columns from features
TARGET_CAT    = "attack_cat"
TARGET_BINARY = "label"
FEATURE_COLS  = [c for c in train.columns if c not in [TARGET_CAT, TARGET_BINARY]]
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── 6. Encode known string columns (proto, service, state) ───────────────────
# These columns contain strings like 'tcp', 'udp', 'CON' etc.
# We force-check all columns for string content regardless of dtype
KNOWN_CAT_COLS = ["proto", "service", "state"]
encoded_cols = []

for col in KNOWN_CAT_COLS:
    if col in FEATURE_COLS:
        # Convert to string first to handle any mixed types
        train[col] = train[col].astype(str).str.strip()
        test[col]  = test[col].astype(str).str.strip()

        le_temp = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0)
        le_temp.fit(combined)
        train[col] = le_temp.transform(train[col])
        test[col]  = le_temp.transform(test[col])
        encoded_cols.append(col)

if encoded_cols:
    print(f"  Encoded string columns: {encoded_cols}")
else:
    print("  No string columns found in features.")

# ── 7. Encode attack_cat → integer ───────────────────────────────────────────
le = LabelEncoder()
all_cats = pd.concat([train[TARGET_CAT], test[TARGET_CAT]], axis=0)
le.fit(all_cats)

train["attack_cat_encoded"] = le.transform(train[TARGET_CAT])
test["attack_cat_encoded"]  = le.transform(test[TARGET_CAT])

# Save label mapping → used by backend to reverse integer → attack name
label_map = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
with open(os.path.join(MODEL_DIR, "label_encoder_classes.json"), "w") as f:
    json.dump(label_map, f, indent=2)
print(f"Label mapping saved: {label_map}")

# ── 8. Normalize numeric features (MinMaxScaler) ──────────────────────────────
scaler = MinMaxScaler()
train[FEATURE_COLS] = scaler.fit_transform(train[FEATURE_COLS])
test[FEATURE_COLS]  = scaler.transform(test[FEATURE_COLS])

# Save scaler params → used by backend to normalize incoming predictions
scaler_params = {
    "feature_cols": FEATURE_COLS,
    "data_min":     scaler.data_min_.tolist(),
    "data_max":     scaler.data_max_.tolist(),
    "scale":        scaler.scale_.tolist(),
    "min":          scaler.min_.tolist(),
}
with open(os.path.join(MODEL_DIR, "scaler_params.json"), "w") as f:
    json.dump(scaler_params, f, indent=2)
print("Scaler params saved.")

# ── 9. Save processed data ────────────────────────────────────────────────────
train.to_parquet(os.path.join(OUT_DIR, "unsw_train.parquet"), index=False)
test.to_parquet(os.path.join(OUT_DIR,  "unsw_test.parquet"),  index=False)
print(f"Saved → data/processed/unsw_train.parquet  ({len(train):,} rows)")
print(f"Saved → data/processed/unsw_test.parquet   ({len(test):,} rows)")

# ── 10. Final summary ─────────────────────────────────────────────────────────
print()
print("=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)
print(f"Train rows (after dedup) : {len(train):,}")
print(f"Test rows                : {len(test):,}")
print(f"Feature columns          : {len(FEATURE_COLS)}")
print(f"Attack categories        : {len(le.classes_)}")
print()
print("Class distribution (train after dedup):")
print(train[TARGET_CAT].value_counts())
print()
print("Files saved:")
print("  data/processed/unsw_train.parquet")
print("  data/processed/unsw_test.parquet")
print("  ml_pipeline/models/label_encoder_classes.json")
print("  ml_pipeline/models/scaler_params.json")