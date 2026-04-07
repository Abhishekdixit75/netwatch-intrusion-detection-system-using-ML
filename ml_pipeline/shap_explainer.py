"""
ml_pipeline/shap_explainer.py

Fits a SHAP TreeExplainer on the trained Random Forest and saves:
  - ml_pipeline/models/shap_explainer.pkl
  - data/processed/shap_summary.png
  - data/processed/shap_beeswarm.png
  - data/processed/shap_per_class/shap_class_{name}.png
  - ml_pipeline/models/shap_sample_values.json  (for dashboard preview)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_PATH   = "data/processed/unsw_test_selected.parquet"
MODEL_DIR   = "ml_pipeline/models"
OUT_DIR     = "data/processed"
FEAT_PATH   = os.path.join(MODEL_DIR, "selected_features.json")
LABEL_PATH  = os.path.join(MODEL_DIR, "label_encoder_classes.json")
SHAP_OUTDIR = os.path.join(OUT_DIR, "shap_per_class")

os.makedirs(SHAP_OUTDIR, exist_ok=True)

# ── 1. Load data and model ────────────────────────────────────────────────────
print("Loading data and model...")
test = pd.read_parquet(TEST_PATH)

with open(FEAT_PATH)  as f: feat_info = json.load(f)
with open(LABEL_PATH) as f: label_map = json.load(f)

FEATURE_COLS = feat_info["selected_features"]
CLASS_NAMES  = [label_map[str(i)] for i in range(len(label_map))]
TARGET       = "attack_cat_encoded"

rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
print("  Model loaded.")

# ── 2. Sample data for SHAP (SHAP is slow on full test set) ──────────────────
# Use 500 samples — enough for meaningful explanations, fast enough to run
np.random.seed(42)
sample_idx  = np.random.choice(len(test), size=500, replace=False)
X_sample    = test[FEATURE_COLS].iloc[sample_idx].values
y_sample    = test[TARGET].iloc[sample_idx].values
df_sample   = test[FEATURE_COLS].iloc[sample_idx]

print(f"  Using {len(X_sample)} samples for SHAP computation.")

# ── 3. Fit SHAP TreeExplainer ─────────────────────────────────────────────────
print("\nFitting SHAP TreeExplainer...")
explainer = shap.TreeExplainer(rf)
print("  Explainer fitted.")

# ── 4. Compute SHAP values ────────────────────────────────────────────────────
print("Computing SHAP values (this may take ~1-2 minutes)...")
shap_raw = explainer.shap_values(X_sample)

# ── 5. Standardize SHAP array to (n_classes, n_samples, n_features) ──────────
shap_array = np.array(shap_raw)
print(f"  Raw SHAP shape: {shap_array.shape}")

# If shape is (samples, features, classes), transpose to (classes, samples, features)
if shap_array.ndim == 3 and shap_array.shape[2] == len(CLASS_NAMES) and shap_array.shape[0] != len(CLASS_NAMES):
    shap_array = np.transpose(shap_array, (2, 0, 1))
    print(f"  Standardized SHAP shape: {shap_array.shape}")

# ── 6. Save explainer ─────────────────────────────────────────────────────────
joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
print("Saved → ml_pipeline/models/shap_explainer.pkl")

# ── 7. Summary plot — mean absolute SHAP across all classes ──────────────────
print("\nGenerating SHAP summary plot...")

# Mean absolute SHAP value per feature across all classes and samples
# Shape: (n_features,)
mean_abs_shap = np.abs(shap_array).mean(axis=(0, 1))

feat_importance_df = pd.DataFrame({
    "feature":   FEATURE_COLS,
    "shap_mean": mean_abs_shap
}).sort_values("shap_mean", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(feat_importance_df["feature"], feat_importance_df["shap_mean"],
               color="#2E75B6", edgecolor="white")
ax.set_title("SHAP Feature Importance — Mean |SHAP| Across All Classes", fontsize=13)
ax.set_xlabel("Mean |SHAP Value|")

# Add value labels
for bar, val in zip(bars, feat_importance_df["shap_mean"]):
    ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)

plt.tight_layout()
path = os.path.join(OUT_DIR, "shap_summary.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {path}")

# ── 8. SHAP beeswarm plot (built-in shap plot) ────────────────────────────────
print("Generating SHAP beeswarm plot...")

# Use class 6 (Normal) vs rest for the beeswarm — most interpretable
try:
    normal_idx = list(label_map.values()).index("Normal")
except ValueError:
    normal_idx = 0

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_array[normal_idx],
    df_sample,
    feature_names=FEATURE_COLS,
    show=False,
    plot_type="dot",
    max_display=20
)
plt.title("SHAP Beeswarm — Normal vs Attack", fontsize=13)
plt.tight_layout()
path = os.path.join(OUT_DIR, "shap_beeswarm.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {path}")

# ── 9. Per-class SHAP bar plots ───────────────────────────────────────────────
print("Generating per-class SHAP plots...")

for class_idx, class_name in enumerate(CLASS_NAMES):
    sv = shap_array[class_idx]            # (n_samples, n_features)
    mean_abs = np.abs(sv).mean(axis=0)    # (n_features,)

    feat_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "shap":    mean_abs
    }).sort_values("shap", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feat_df["feature"], feat_df["shap"], color="#C55A11", edgecolor="white")
    ax.set_title(f"SHAP — Top Features for: {class_name}", fontsize=12)
    ax.set_xlabel("Mean |SHAP Value|")
    plt.tight_layout()

    fname = f"shap_class_{class_name.lower().replace(' ', '_')}.png"
    path  = os.path.join(SHAP_OUTDIR, fname)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()

print(f"Saved {len(CLASS_NAMES)} per-class SHAP plots → {SHAP_OUTDIR}/")

# ── 10. Save sample SHAP values for dashboard ─────────────────────────────────
# Dashboard SHAP page needs to show explanations for individual predictions
# We precompute and save 20 sample predictions with their SHAP values

print("\nSaving sample SHAP values for dashboard...")

dashboard_samples = []
for i in range(min(20, len(X_sample))):
    sample_shap = {
        CLASS_NAMES[c]: {
            feat: float(shap_array[c][i][j])
            for j, feat in enumerate(FEATURE_COLS)
        }
        for c in range(len(CLASS_NAMES))
    }

    pred_class   = int(rf.predict(X_sample[i:i+1])[0])
    pred_proba   = rf.predict_proba(X_sample[i:i+1])[0]
    true_class   = int(y_sample[i])

    dashboard_samples.append({
        "sample_id":        i,
        "true_class":       CLASS_NAMES[true_class],
        "predicted_class":  CLASS_NAMES[pred_class],
        "confidence":       round(float(pred_proba[pred_class]), 4),
        "correct":          pred_class == true_class,
        "feature_values":   {feat: float(X_sample[i][j]) for j, feat in enumerate(FEATURE_COLS)},
        "shap_for_predicted_class": {
            feat: float(shap_array[pred_class][i][j])
            for j, feat in enumerate(FEATURE_COLS)
        }
    })

with open(os.path.join(MODEL_DIR, "shap_sample_values.json"), "w") as f:
    json.dump(dashboard_samples, f, indent=2)
print("Saved → ml_pipeline/models/shap_sample_values.json")

# ── 10. Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("SHAP COMPLETE")
print("=" * 50)
print(f"Explainer     : saved as shap_explainer.pkl")
print(f"Samples used  : {len(X_sample)}")
print(f"Classes       : {len(CLASS_NAMES)}")
print()
print("Top 5 most important features (global):")
top5 = feat_importance_df.sort_values("shap_mean", ascending=False).head(5)
for _, row in top5.iterrows():
    print(f"  {row['feature']:<25} {row['shap_mean']:.4f}")
print()
print("Files saved:")
print("  ml_pipeline/models/shap_explainer.pkl")
print("  ml_pipeline/models/shap_sample_values.json")
print("  data/processed/shap_summary.png")
print("  data/processed/shap_beeswarm.png")
print(f"  data/processed/shap_per_class/  ({len(CLASS_NAMES)} plots)")
print()
print("Next step → build the backend")