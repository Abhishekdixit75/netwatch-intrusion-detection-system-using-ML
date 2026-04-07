import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.database import init_db, SessionLocal, Prediction, Alert, IPReputation
from backend.predict import run_prediction, SELECTED_FEATURES
from backend.severity import compute_severity
from backend.ip_tracker import record_alert, get_alert_count
from backend.alerts import create_alert

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    init_db()
    yield
    # Cleanup (if any) could go here

app = FastAPI(title="Netwatch IDS", lifespan=lifespan)

class PredictRequest(BaseModel):
    source_ip: str = "0.0.0.0"
    features: dict

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Unified endpoint to score traffic, log predictions, and trigger alerts.
    """
    try:
        # 1. Run inference
        result = run_prediction(req.features)
        
        # 2. Get contextual info
        alert_count = get_alert_count(req.source_ip)
        severity = compute_severity(result["prediction"], result["attack_type"], alert_count)

        # 3. Log Prediction to Database
        db = SessionLocal()
        try:
            pred_row = Prediction(
                source_ip=req.source_ip,
                features_json=json.dumps(req.features),
                prediction=result["prediction"],
                attack_type=result["attack_type"],
                severity=severity,
                anomaly_score=result["anomaly_score"],
                shap_json=json.dumps(result["shap_values"]),
            )
            db.add(pred_row)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"DB Error (Prediction): {e}")
        finally:
            db.close()

        # 4. If attack detected, trigger alert and update reputation
        if result["prediction"] != "Normal":
            record_alert(req.source_ip)
            create_alert(
                source_ip=req.source_ip,
                attack_type=result["attack_type"],
                severity=severity,
                message=f"{result['attack_type']} attack detected from {req.source_ip} (Severity: {severity})"
            )

        return {
            **result,
            "severity": severity,
            "source_ip": req.source_ip,
            "alert_count_historical": alert_count + (1 if result["prediction"] != "Normal" else 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")

@app.get("/alerts")
def get_alerts(limit: int = 50):
    """Returns the most recent intrusion alerts."""
    db = SessionLocal()
    try:
        alerts = db.query(Alert).order_by(Alert.id.desc()).limit(limit).all()
        return [
            {
                "id": a.id,
                "timestamp": a.timestamp.isoformat(),
                "source_ip": a.source_ip,
                "attack_type": a.attack_type,
                "severity": a.severity,
                "message": a.message
            }
            for a in alerts
        ]
    finally:
        db.close()

@app.get("/ip-reputation")
def get_ip_reputation():
    """Returns a list of tracked IPs and their current status."""
    db = SessionLocal()
    try:
        ips = db.query(IPReputation).order_by(IPReputation.alert_count.desc()).all()
        return [
            {
                "ip": r.ip,
                "alert_count": r.alert_count,
                "first_seen": r.first_seen.isoformat() if r.first_seen else None,
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                "status": r.status
            }
            for r in ips
        ]
    finally:
        db.close()

@app.get("/stats")
def get_stats():
    """Returns high-level system statistics (standardized for dashboard)."""
    db = SessionLocal()
    try:
        preds = db.query(Prediction).all()
        total = len(preds)
        normal = sum(1 for p in preds if p.prediction == "Normal")
        attacks = total - normal
        
        by_type = {}
        for p in preds:
            if p.prediction != "Normal":
                by_type[p.attack_type] = by_type.get(p.attack_type, 0) + 1
        
        return {
            "total": total,
            "normal": normal,
            "attacks": attacks,
            "by_type": by_type
        }
    finally:
        db.close()

@app.get("/predictions")
def get_predictions(limit: int = 50):
    """Returns a list of recent predictions for the explainability view."""
    db = SessionLocal()
    try:
        preds = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).all()
        return [
            {
                "id": p.id,
                "timestamp": p.timestamp.isoformat(),
                "source_ip": p.source_ip,
                "prediction": p.prediction,
                "attack_type": p.attack_type,
                "severity": p.severity
            }
            for p in preds
        ]
    finally:
        db.close()

@app.get("/prediction/{pred_id}")
def get_prediction_detail(pred_id: int):
    """Returns full details including SHAP JSON for a specific prediction."""
    db = SessionLocal()
    try:
        p = db.query(Prediction).filter(Prediction.id == pred_id).first()
        if not p:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return {
            "id": p.id,
            "timestamp": p.timestamp.isoformat(),
            "source_ip": p.source_ip,
            "prediction": p.prediction,
            "attack_type": p.attack_type,
            "severity": p.severity,
            "anomaly_score": p.anomaly_score,
            "shap_values": json.loads(p.shap_json) if p.shap_json else {}
        }
    finally:
        db.close()

@app.get("/model-comparison")
def model_comparison():
    """Returns the performance metrics generated during evaluation."""
    try:
        path = "ml_pipeline/models/evaluation_results.json"
        if not os.path.exists(path):
            return {"error": "Evaluation results not found. Run evaluation script first."}
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    """Basic health check for system status."""
    return {"status": "online", "models": "active"}
