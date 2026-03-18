"""
ml_pipeline/train.py

Trains all 4 models on UNSW-NB15 selected features and saves:
  - ml_pipeline/models/rf_model.pkl
  - ml_pipeline/models/iso_model.pkl
  - ml_pipeline/models/xgb_model.pkl
  - ml_pipeline/models/svm_model.pkl
  - ml_pipeline/models/training_meta.json
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH = "data/processed/unsw_train_selected.parquet"
MODEL_DIR  = "ml_pipeline/models"
FEAT_PATH  = os.path.join(MODEL_DIR, "selected_features.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)

with open(FEAT_PATH) as f:
    feat_info = json.load(f)

FEATURE_COLS = feat_info["selected_features"]
TARGET       = "attack_cat_encoded"

X_train = train[FEATURE_COLS].values
y_train = train[TARGET].values

print(f"  Samples  : {len(X_train):,}")
print(f"  Features : {len(FEATURE_COLS)}")
print(f"  Classes  : {np.unique(y_train)}")

# ── Helper ────────────────────────────────────────────────────────────────────
def train_model(name, model, X, y=None):
    print(f"\nTraining {name}...")
    t0 = time.time()
    if y is not None:
        model.fit(X, y)
        preds = model.predict(X)
        acc   = accuracy_score(y, preds)
    else:
        # Isolation Forest — unsupervised, no y
        model.fit(X)
        acc = None
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s" + (f"  |  Train accuracy: {acc*100:.2f}%" if acc else "  |  Unsupervised"))
    return model, elapsed, acc

meta = {}

# ── 2. Random Forest (Primary Classifier) ─────────────────────────────────────
rf, rf_time, rf_acc = train_model(
    "Random Forest",
    RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",  # handles class imbalance (Worms=130 samples)
        n_jobs=-1,
        random_state=42
    ),
    X_train, y_train
)
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
print("  Saved → ml_pipeline/models/rf_model.pkl")
meta["random_forest"] = {"train_time_sec": round(rf_time, 2), "train_accuracy": round(rf_acc, 4)}

# ── 3. Isolation Forest (Anomaly Detector) ────────────────────────────────────
# Trained on NORMAL traffic only — flags anything that deviates as anomalous
X_normal = train[train["label"] == 0][FEATURE_COLS].values
print(f"\nTraining Isolation Forest on {len(X_normal):,} normal samples only...")

iso, iso_time, _ = train_model(
    "Isolation Forest",
    IsolationForest(
        n_estimators=100,
        contamination=0.05,   # expect ~5% anomalies in unseen traffic
        n_jobs=-1,
        random_state=42
    ),
    X_normal
)
joblib.dump(iso, os.path.join(MODEL_DIR, "iso_model.pkl"))
print("  Saved → ml_pipeline/models/iso_model.pkl")
meta["isolation_forest"] = {"train_time_sec": round(iso_time, 2), "train_accuracy": None, "note": "Trained on normal traffic only"}

# ── 4. XGBoost (Comparison Model) ─────────────────────────────────────────────
# Compute sample weights manually for XGBoost (no class_weight param)
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

xgb, xgb_time, xgb_acc = train_model(
    "XGBoost",
    XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0
    ),
    X_train, y_train
)
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))
print("  Saved → ml_pipeline/models/xgb_model.pkl")
meta["xgboost"] = {"train_time_sec": round(xgb_time, 2), "train_accuracy": round(xgb_acc, 4)}

# ── 5. SVM (Comparison Model) ─────────────────────────────────────────────────
# SVM is slow on large datasets — subsample to 20k for training
print("\nSVM: subsampling to 20,000 rows (SVM does not scale to full dataset)...")
np.random.seed(42)
idx = np.random.choice(len(X_train), size=min(20000, len(X_train)), replace=False)
X_svm = X_train[idx]
y_svm = y_train[idx]

svm, svm_time, svm_acc = train_model(
    "SVM",
    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        decision_function_shape="ovr",
        random_state=42
    ),
    X_svm, y_svm
)
joblib.dump(svm, os.path.join(MODEL_DIR, "svm_model.pkl"))
print("  Saved → ml_pipeline/models/svm_model.pkl")
meta["svm"] = {
    "train_time_sec": round(svm_time, 2),
    "train_accuracy": round(svm_acc, 4),
    "note": f"Trained on {len(X_svm):,} subsampled rows"
}

# ── 6. Save training metadata ─────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "training_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("\nSaved → ml_pipeline/models/training_meta.json")

# ── 7. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
for model_name, info in meta.items():
    acc_str = f"{info['train_accuracy']*100:.2f}%" if info.get("train_accuracy") else "N/A (unsupervised)"
    print(f"  {model_name:<20} time: {info['train_time_sec']}s   train_acc: {acc_str}")
print()
print("Models saved:")
print("  ml_pipeline/models/rf_model.pkl")
print("  ml_pipeline/models/iso_model.pkl")
print("  ml_pipeline/models/xgb_model.pkl")
print("  ml_pipeline/models/svm_model.pkl")
print()
print("Next step → run ml_pipeline/evaluate.py")