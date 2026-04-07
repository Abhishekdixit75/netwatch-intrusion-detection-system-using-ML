import datetime
from backend.database import SessionLocal, Alert

def create_alert(source_ip: str, attack_type: str, severity: str, message: str):
    """
    Records a high-level security alert in the database.
    """
    db = SessionLocal()
    try:
        alert = Alert(
            timestamp=datetime.datetime.now(datetime.UTC),
            source_ip=source_ip,
            attack_type=attack_type,
            severity=severity,
            message=message
        )
        db.add(alert)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error creating alert: {e}")
    finally:
        db.close()
