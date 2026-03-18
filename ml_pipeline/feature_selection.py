"""
ml_pipeline/feature_selection.py

Selects the most important features using Random Forest importance scores.
Saves:
  - ml_pipeline/models/selected_features.json
  - data/processed/unsw_train_selected.parquet
  - data/processed/unsw_test_selected.parquet
  - data/processed/feature_importance.png
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH   = "data/processed/unsw_train.parquet"
TEST_PATH    = "data/processed/unsw_test.parquet"
MODEL_DIR    = "ml_pipeline/models"
OUT_DIR      = "data/processed"

os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load processed data ────────────────────────────────────────────────────
print("Loading processed data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
print(f"  Train: {train.shape}  |  Test: {test.shape}")

# ── 2. Define features and target ─────────────────────────────────────────────
DROP_COLS   = ["attack_cat", "attack_cat_encoded", "label"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
TARGET       = "attack_cat_encoded"

X_train = train[FEATURE_COLS]
y_train = train[TARGET]

print(f"Features going in: {len(FEATURE_COLS)}")
print(f"Training samples : {len(X_train):,}")

# ── 3. Train a quick Random Forest to get importance scores ───────────────────
print("\nFitting Random Forest for feature importance...")
rf = RandomForestClassifier(
    n_estimators=100,       # quick run — not the final model
    max_depth=15,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
print("  Done.")

# ── 4. Extract and rank importances ───────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)

print("\nFeature importances (all):")
for feat, imp in importances.items():
    print(f"  {feat:<30} {imp:.4f}")

# ── 5. Select top features ────────────────────────────────────────────────────
# Strategy: keep features that cumulatively explain 95% of importance
# This is more principled than a fixed cutoff
cumulative = importances.cumsum() / importances.sum()
selected = cumulative[cumulative <= 0.95].index.tolist()

# Always keep at least the top feature
if len(selected) == 0:
    selected = [importances.index[0]]

# Also ensure we have at least 10 features for model robustness
if len(selected) < 10:
    selected = importances.index[:10].tolist()

print(f"\nSelected {len(selected)} features (covering 95% of cumulative importance):")
for f in selected:
    print(f"  {f:<30} importance: {importances[f]:.4f}")

# ── 6. Save selected feature list ─────────────────────────────────────────────
feature_info = {
    "selected_features": selected,
    "feature_importances": {f: float(importances[f]) for f in selected},
    "total_features_before": len(FEATURE_COLS),
    "total_features_after": len(selected),
    "cumulative_importance_covered": float(importances[selected].sum() / importances.sum())
}

with open(os.path.join(MODEL_DIR, "selected_features.json"), "w") as f:
    json.dump(feature_info, f, indent=2)
print(f"\nSaved → ml_pipeline/models/selected_features.json")

# ── 7. Apply selection to train and test ──────────────────────────────────────
KEEP_COLS = selected + ["attack_cat", "attack_cat_encoded", "label"]

train_selected = train[KEEP_COLS]
test_selected  = test[KEEP_COLS]

train_selected.to_parquet(os.path.join(OUT_DIR, "unsw_train_selected.parquet"), index=False)
test_selected.to_parquet(os.path.join(OUT_DIR,  "unsw_test_selected.parquet"),  index=False)
print(f"Saved → data/processed/unsw_train_selected.parquet  ({len(train_selected):,} rows)")
print(f"Saved → data/processed/unsw_test_selected.parquet   ({len(test_selected):,} rows)")

# ── 8. Plot feature importances ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left — all features
colors_all = ["#2E75B6" if f in selected else "#BFBFBF" for f in importances.index]
axes[0].barh(importances.index[::-1], importances.values[::-1], color=colors_all[::-1])
axes[0].set_title("All Features — Importance Scores\n(blue = selected, grey = dropped)", fontsize=12)
axes[0].set_xlabel("Importance")
axes[0].axvline(x=importances[selected[-1]], color="red", linestyle="--", alpha=0.6, label="Selection threshold")
axes[0].legend()

# Right — selected features only
sel_imp = importances[selected]
axes[1].barh(sel_imp.index[::-1], sel_imp.values[::-1], color="#2E75B6")
axes[1].set_title(f"Selected {len(selected)} Features", fontsize=12)
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved → data/processed/feature_importance.png")

# ── 9. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("FEATURE SELECTION COMPLETE")
print("=" * 50)
print(f"Features before : {len(FEATURE_COLS)}")
print(f"Features after  : {len(selected)}")
print(f"Importance covered: {feature_info['cumulative_importance_covered']*100:.1f}%")
print(f"Selected        : {selected}")