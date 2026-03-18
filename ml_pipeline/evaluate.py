"""
ml_pipeline/evaluate.py

Evaluates all trained models on the test set and saves:
  - ml_pipeline/models/evaluation_results.json
  - data/processed/confusion_matrix_rf.png
  - data/processed/confusion_matrix_xgb.png
  - data/processed/roc_curve.png
  - data/processed/model_comparison.png
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report
)

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_PATH  = "data/processed/unsw_test_selected.parquet"
MODEL_DIR  = "ml_pipeline/models"
OUT_DIR    = "data/processed"
FEAT_PATH  = os.path.join(MODEL_DIR, "selected_features.json")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder_classes.json")

# ── 1. Load test data ─────────────────────────────────────────────────────────
print("Loading test data...")
test = pd.read_parquet(TEST_PATH)

with open(FEAT_PATH)  as f: feat_info  = json.load(f)
with open(LABEL_PATH) as f: label_map  = json.load(f)

FEATURE_COLS = feat_info["selected_features"]
TARGET       = "attack_cat_encoded"
CLASS_NAMES  = [label_map[str(i)] for i in range(len(label_map))]

X_test = test[FEATURE_COLS].values
y_test = test[TARGET].values

print(f"  Test samples : {len(X_test):,}")
print(f"  Classes      : {CLASS_NAMES}")

# ── 2. Load models ────────────────────────────────────────────────────────────
print("\nLoading models...")
rf  = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
iso = joblib.load(os.path.join(MODEL_DIR, "iso_model.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
svm = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
print("  All models loaded.")

# ── Helper: evaluate a classifier ────────────────────────────────────────────
def evaluate_classifier(name, model, X, y, class_names):
    t0    = time.time()
    preds = model.predict(X)
    infer = (time.time() - t0) / len(X) * 1000  # ms per sample

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, average="weighted", zero_division=0)
    rec  = recall_score(y, preds, average="weighted", zero_division=0)
    f1   = f1_score(y, preds, average="weighted", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(y, preds, average=None, zero_division=0)
    per_class    = {class_names[i]: round(float(per_class_f1[i]), 4)
                    for i in range(len(class_names))}

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"  Inference : {infer:.4f} ms/sample")
    print(f"\n  Per-class F1:")
    for cls, score in per_class.items():
        bar = "█" * int(score * 20)
        print(f"    {cls:<18} {score:.4f}  {bar}")
    print(f"\n{classification_report(y, preds, target_names=class_names, zero_division=0)}")

    return {
        "accuracy":          round(acc,  4),
        "precision":         round(prec, 4),
        "recall":            round(rec,  4),
        "f1_score":          round(f1,   4),
        "inference_ms":      round(infer, 6),
        "per_class_f1":      per_class,
        "predictions":       preds
    }

# ── 3. Evaluate classifiers ───────────────────────────────────────────────────
results = {}

rf_res  = evaluate_classifier("Random Forest", rf,  X_test, y_test, CLASS_NAMES)
xgb_res = evaluate_classifier("XGBoost",       xgb, X_test, y_test, CLASS_NAMES)
svm_res = evaluate_classifier("SVM",           svm, X_test, y_test, CLASS_NAMES)

results["random_forest"] = {k: v for k, v in rf_res.items()  if k != "predictions"}
results["xgboost"]       = {k: v for k, v in xgb_res.items() if k != "predictions"}
results["svm"]           = {k: v for k, v in svm_res.items() if k != "predictions"}

# ── 4. Isolation Forest evaluation ───────────────────────────────────────────
print(f"\n{'='*50}")
print("Isolation Forest (Anomaly Detection)")
print(f"{'='*50}")

t0            = time.time()
iso_preds_raw = iso.predict(X_test)        # 1 = normal, -1 = anomaly
iso_scores    = iso.decision_function(X_test)  # higher = more normal
infer_iso     = (time.time() - t0) / len(X_test) * 1000

# Convert to binary: 1 = anomaly, 0 = normal (align with label column)
iso_binary = (iso_preds_raw == -1).astype(int)
y_binary   = (y_test != 6).astype(int)    # 6 = Normal class index

iso_acc  = accuracy_score(y_binary, iso_binary)
iso_prec = precision_score(y_binary, iso_binary, zero_division=0)
iso_rec  = recall_score(y_binary, iso_binary, zero_division=0)
iso_f1   = f1_score(y_binary, iso_binary, zero_division=0)

print(f"  (Binary evaluation: Normal vs Any Attack)")
print(f"  Accuracy  : {iso_acc*100:.2f}%")
print(f"  Precision : {iso_prec*100:.2f}%")
print(f"  Recall    : {iso_rec*100:.2f}%")
print(f"  F1 Score  : {iso_f1*100:.2f}%")
print(f"  Inference : {infer_iso:.4f} ms/sample")

results["isolation_forest"] = {
    "accuracy":     round(iso_acc,  4),
    "precision":    round(iso_prec, 4),
    "recall":       round(iso_rec,  4),
    "f1_score":     round(iso_f1,   4),
    "inference_ms": round(infer_iso, 6),
    "note":         "Binary evaluation — Normal vs Attack"
}

# ── 5. Save evaluation results ────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → ml_pipeline/models/evaluation_results.json")

# ── 6. Confusion matrices ─────────────────────────────────────────────────────
def plot_confusion_matrix(name, preds, y_true, class_names, filename):
    cm = confusion_matrix(y_true, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{name} — Confusion Matrix (Counts)", fontsize=13)
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    axes[0].tick_params(axis="x", rotation=45)

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{name} — Confusion Matrix (%)", fontsize=13)
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

plot_confusion_matrix("Random Forest", rf_res["predictions"],  y_test, CLASS_NAMES, "confusion_matrix_rf.png")
plot_confusion_matrix("XGBoost",       xgb_res["predictions"], y_test, CLASS_NAMES, "confusion_matrix_xgb.png")
plot_confusion_matrix("SVM",           svm_res["predictions"], y_test, CLASS_NAMES, "confusion_matrix_svm.png")

# ── 7. Model comparison bar chart ─────────────────────────────────────────────
models     = ["Random Forest", "XGBoost", "SVM", "Isolation Forest"]
accuracies = [results["random_forest"]["accuracy"],  results["xgboost"]["accuracy"],
              results["svm"]["accuracy"],             results["isolation_forest"]["accuracy"]]
f1_scores  = [results["random_forest"]["f1_score"],  results["xgboost"]["f1_score"],
              results["svm"]["f1_score"],             results["isolation_forest"]["f1_score"]]
infer_ms   = [results["random_forest"]["inference_ms"], results["xgboost"]["inference_ms"],
              results["svm"]["inference_ms"],            results["isolation_forest"]["inference_ms"]]

x     = np.arange(len(models))
width = 0.35
colors_acc = ["#2E75B6", "#2E75B6", "#2E75B6", "#70AD47"]
colors_f1  = ["#1F4E79", "#1F4E79", "#1F4E79", "#375623"]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Accuracy
bars = axes[0].bar(x, [a*100 for a in accuracies], color=colors_acc, edgecolor="white", width=0.6)
axes[0].set_title("Accuracy (%)", fontsize=13)
axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15, ha="right")
axes[0].set_ylim(0, 110)
axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
for bar, val in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10)

# F1 Score
bars = axes[1].bar(x, [f*100 for f in f1_scores], color=colors_f1, edgecolor="white", width=0.6)
axes[1].set_title("F1 Score (Weighted %)", fontsize=13)
axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15, ha="right")
axes[1].set_ylim(0, 110)
axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
for bar, val in zip(bars, f1_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10)

# Inference time
bars = axes[2].bar(x, infer_ms, color=["#C55A11"]*4, edgecolor="white", width=0.6)
axes[2].set_title("Inference Time (ms/sample)", fontsize=13)
axes[2].set_xticks(x); axes[2].set_xticklabels(models, rotation=15, ha="right")
for bar, val in zip(bars, infer_ms):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Model Comparison — UNSW-NB15 Test Set", fontsize=15, y=1.02)
plt.tight_layout()
path = os.path.join(OUT_DIR, "model_comparison.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {path}")

# ── 8. Per-class F1 heatmap ───────────────────────────────────────────────────
per_class_df = pd.DataFrame({
    "Random Forest": rf_res["per_class_f1"],
    "XGBoost":       xgb_res["per_class_f1"],
    "SVM":           svm_res["per_class_f1"],
})

plt.figure(figsize=(10, 7))
sns.heatmap(per_class_df, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=0.5, vmin=0, vmax=1)
plt.title("Per-Class F1 Score by Model", fontsize=13)
plt.tight_layout()
path = os.path.join(OUT_DIR, "per_class_f1.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {path}")

# ── 9. Final summary ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print(f"{'Model':<22} {'Accuracy':>10} {'F1':>10} {'Inference':>14}")
print("-" * 60)
for m, key in [("Random Forest","random_forest"),("XGBoost","xgboost"),
               ("SVM","svm"),("Isolation Forest","isolation_forest")]:
    r = results[key]
    print(f"  {m:<20} {r['accuracy']*100:>9.2f}% {r['f1_score']*100:>9.2f}%  {r['inference_ms']:>10.4f}ms")
print()
print("Files saved:")
print("  ml_pipeline/models/evaluation_results.json")
print("  data/processed/confusion_matrix_rf.png")
print("  data/processed/confusion_matrix_xgb.png")
print("  data/processed/confusion_matrix_svm.png")
print("  data/processed/model_comparison.png")
print("  data/processed/per_class_f1.png")
print()
print("Next step → run ml_pipeline/shap_explainer.py")