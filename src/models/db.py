from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Detection(db.Model):
    __tablename__ = 'detections'
    id = db.Column(db.Integer, primary_key=True)
    rel_path = db.Column(db.String(255), nullable=False)      # ruta relativa bajo /static
    pred_label = db.Column(db.String(64), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    prob_anthracnose = db.Column(db.Float, nullable=False)
    prob_healthy = db.Column(db.Float, nullable=False)
    real_label = db.Column(db.String(64), nullable=True)  # Etiqueta real (ground truth)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def init_db():
    db.create_all()
