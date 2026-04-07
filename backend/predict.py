import json
import os
import numpy as np
import joblib
import shap

# Configuration Paths
MODEL_DIR = "ml_pipeline/models"
RF_PATH    = os.path.join(MODEL_DIR, "rf_model.pkl")
ISO_PATH   = os.path.join(MODEL_DIR, "iso_model.pkl")
SHAP_PATH  = os.path.join(MODEL_DIR, "shap_explainer.pkl")
FEAT_PATH  = os.path.join(MODEL_DIR, "selected_features.json")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder_classes.json")
SCALE_PATH = os.path.join(MODEL_DIR, "scaler_params.json")

# Load models and metadata at startup
print("Loading models and metadata for backend inference...")
rf_model  = joblib.load(RF_PATH)
iso_model = joblib.load(ISO_PATH)
explainer = joblib.load(SHAP_PATH)

with open(FEAT_PATH) as f:
    feat_data = json.load(f)
    SELECTED_FEATURES = feat_data["selected_features"]

with open(LABEL_PATH) as f:
    # JSON keys are always strings, so we keep as is for .get(str(idx))
    LABEL_CLASSES = json.load(f)

with open(SCALE_PATH) as f:
    SCALER_PARAMS = json.load(f)

# Load categorical mappings (proto, service, state) for raw string lookup
MAPPING_PATH = os.path.join(MODEL_DIR, "category_mappings.json")
if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH) as f:
        CAT_MAPPINGS = json.load(f)
else:
    CAT_MAPPINGS = {}

def run_prediction(features_dict: dict) -> dict:
    """
    Processes raw features, runs inference, and generates SHAP explanations.
    """
    # 1. Build and scale feature vector in the correct order
    raw_vector = []
    for feat in SELECTED_FEATURES:
        val = features_dict.get(feat, 0.0)
        
        # If it's a categorical feature (proto, service, state) and we got a string, map it
        if feat in CAT_MAPPINGS and isinstance(val, (str, bytes)):
            # Convert to string and strip to match mapping keys
            val_str = str(val).strip()
            val = CAT_MAPPINGS[feat].get(val_str, 0)
        
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = 0.0
        
        # Apply scaling: X_scaled = X * scale + min
        if feat in SCALER_PARAMS["feature_cols"]:
            idx = SCALER_PARAMS["feature_cols"].index(feat)
            scale = SCALER_PARAMS["scale"][idx]
            min_offset = SCALER_PARAMS["min"][idx]
            val = val * scale + min_offset
        
        raw_vector.append(val)
    
    vector = np.array([raw_vector])

    # 2. Primary classification (Random Forest)
    pred_class_idx = int(rf_model.predict(vector)[0])
    attack_type    = LABEL_CLASSES.get(str(pred_class_idx), "Unknown")
    prediction     = "Normal" if attack_type == "Normal" else "Attack"

    # 3. Anomaly score (Isolation Forest)
    # Note: decision_function returns signed proximity. Lower = more anomalous.
    anomaly_score = float(iso_model.decision_function(vector)[0])

    # 4. SHAP explanations
    shap_values = explainer.shap_values(vector)
    
    # SHAP output formats vary by model and library version.
    # We normalize to a 1D array of shape (n_features,) for the predicted class.
    if isinstance(shap_values, list):
        # Format: List of arrays [class_0, class_1, ...]
        shap_for_class = shap_values[pred_class_idx][0]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        # Format: (n_samples, n_features, n_classes)
        # We extract the first sample and the index of the predicted class
        shap_for_class = shap_values[0, :, pred_class_idx]
    else:
        # Fallback for binary or simple array (n_samples, n_features)
        shap_for_class = shap_values[0]
        
    shap_dict = {
        feat: round(float(shap_for_class[i]), 4)
        for i, feat in enumerate(SELECTED_FEATURES)
    }

    return {
        "prediction": prediction,
        "attack_type": attack_type,
        "anomaly_score": anomaly_score,
        "shap_values": shap_dict,
    }
