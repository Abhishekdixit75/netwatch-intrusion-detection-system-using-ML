from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

DATABASE_URL = "sqlite:///./ids.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class DetectionBase:
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    source_ip = Column(String)
    attack_type = Column(String)
    severity = Column(String)

class Prediction(Base, DetectionBase):
    __tablename__ = "predictions"
    features_json = Column(Text)
    prediction = Column(String)
    anomaly_score = Column(Float)
    shap_json = Column(Text)

class Alert(Base, DetectionBase):
    __tablename__ = "alerts"
    message = Column(Text)

class IPReputation(Base):
    __tablename__ = "ip_reputation"
    ip = Column(String, primary_key=True)
    alert_count = Column(Integer, default=0)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    status = Column(String, default="watching")

def init_db():
    Base.metadata.create_all(bind=engine)
