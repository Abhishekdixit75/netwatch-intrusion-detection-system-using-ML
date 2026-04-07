"""
ml_pipeline/train.py

Trains and fine-tunes all 4 models on UNSW-NB15 selected features.
Now implements RandomizedSearchCV for hyperparameter optimization to maximize accuracy and F1 metrics.
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
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

# ── Tuning Helper ─────────────────────────────────────────────────────────────
def optimize_model(name, model, grid, X, y, sample_weight=None):
    print(f"\n[Search] Optimizing {name}...")
    t0 = time.time()
    
    # Using 3-fold CV and 10 iterations to balance depth vs speed
    search = RandomizedSearchCV(
        model, 
        param_distributions=grid,
        n_iter=10,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    if sample_weight is not None:
        search.fit(X, y, sample_weight=sample_weight)
    else:
        search.fit(X, y)
        
    elapsed = time.time() - t0
    print(f"  Optimization took {elapsed:.1f}s")
    print(f"  Best params: {search.best_params_}")
    
    # Final check on training set
    best_model = search.best_estimator_
    preds = best_model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='weighted')
    print(f"  Post-tuning Train Acc: {acc*100:.2f}% | F1: {f1:.4f}")
    
    return best_model, elapsed, acc

meta = {}

# ── 2. Random Forest Optimization ─────────────────────────────────────────────
rf_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_best, rf_time, rf_acc = optimize_model("Random Forest", rf_base, rf_grid, X_train, y_train)

joblib.dump(rf_best, os.path.join(MODEL_DIR, "rf_model.pkl"))
meta["random_forest"] = {"train_time_sec": round(rf_time, 2), "train_accuracy": round(rf_acc, 4)}

# ── 3. XGBoost Optimization ───────────────────────────────────────────────────
# Fix for previous bug: XGBoost now uses correctly computed sample weights during search
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

xgb_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'gamma': [0, 0.1, 0.2]
}

xgb_base = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0)
xgb_best, xgb_time, xgb_acc = optimize_model("XGBoost", xgb_base, xgb_grid, X_train, y_train, sample_weight=sample_weights)

joblib.dump(xgb_best, os.path.join(MODEL_DIR, "xgb_model.pkl"))
meta["xgboost"] = {"train_time_sec": round(xgb_time, 2), "train_accuracy": round(xgb_acc, 4)}

# ── 4. Isolation Forest (Fixed Refinement) ────────────────────────────────────
X_normal = train[train["label"] == 0][FEATURE_COLS].values
print(f"\nTraining Isolation Forest on {len(X_normal):,} normal samples...")
t0 = time.time()
iso = IsolationForest(n_estimators=200, contamination=0.03, n_jobs=-1, random_state=42)
iso.fit(X_normal)
iso_time = time.time() - t0
joblib.dump(iso, os.path.join(MODEL_DIR, "iso_model.pkl"))
meta["isolation_forest"] = {"train_time_sec": round(iso_time, 2), "train_accuracy": None, "note": "Fine-tuned contamination to 0.03"}

# ── 5. SVM Optimization (Efficiency Pass) ─────────────────────────────────────
# SVM is still slow; search on 10k but optimize C and Gamma
idx = np.random.choice(len(X_train), size=10000, replace=False)
X_svm_small = X_train[idx]
y_svm_small = y_train[idx]

svm_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 1, 0.1],
    'kernel': ['rbf', 'poly']
}
svm_base = SVC(class_weight="balanced", random_state=42)
svm_best, svm_time, svm_acc = optimize_model("SVM", svm_base, svm_grid, X_svm_small, y_svm_small)

joblib.dump(svm_best, os.path.join(MODEL_DIR, "svm_model.pkl"))
meta["svm"] = {"train_time_sec": round(svm_time, 2), "train_accuracy": round(svm_acc, 4), "note": "Tuned on 10k rows"}

# ── 6. Save training metadata ─────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "training_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "="*50)
print("FINE-TUNING COMPLETE")
print("="*50)