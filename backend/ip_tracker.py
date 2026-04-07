import datetime
from collections import defaultdict
from backend.database import SessionLocal, IPReputation

# In-memory monitoring window: ip -> list of timestamps
_recent_alerts = defaultdict(list)
WINDOW_SECONDS = 60

def record_alert(ip: str):
    """
    Tracks alerts in memory for a sliding window and updates IP reputation in the DB.
    Optimized for demo: Lower thresholds to show auto-blocking in action.
    """
    now = datetime.datetime.now(datetime.UTC)
    _recent_alerts[ip].append(now)
    
    # Prune old entries from memory window
    _recent_alerts[ip] = [
        t for t in _recent_alerts[ip]
        if (now - t).total_seconds() <= WINDOW_SECONDS
    ]
    
    burst_count = len(_recent_alerts[ip])
    
    db = SessionLocal()
    try:
        rep = db.query(IPReputation).filter(IPReputation.ip == ip).first()
        if not rep:
            rep = IPReputation(
                ip=ip, 
                alert_count=0, 
                first_seen=now, 
                last_seen=now,
                status="monitoring"
            )
            db.add(rep)
        
        rep.alert_count += 1
        rep.last_seen = now
        
        # Optimized thresholds for demo feedback
        if burst_count >= 5:
            rep.status = "blocked"
        elif burst_count >= 3:
            rep.status = "watching"
        else:
            # Maintain monitoring status if not escalated
            if not rep.status:
                rep.status = "monitoring"
            
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error updating IP reputation: {e}")
    finally:
        db.close()

def get_alert_count(ip: str) -> int:
    """
    Returns total historical alert count for an IP.
    """
    db = SessionLocal()
    try:
        rep = db.query(IPReputation).filter(IPReputation.ip == ip).first()
        return rep.alert_count if rep else 0
    finally:
        db.close()
