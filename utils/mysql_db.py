"""
MySQL Database Integration for Deteksi PMK
Handles saving and retrieving predictions and diagnosis history
"""

import os
import json
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Database configuration from environment variables
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '3306')
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_NAME = os.environ.get('DB_NAME', 'deteksi_pmk')

# Create database URI
if DB_PASSWORD:
    DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DATABASE_URI = f"mysql+pymysql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine
engine = create_engine(DATABASE_URI, echo=False, pool_pre_ping=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)


class Prediction(Base):
    """Model untuk menyimpan hasil prediksi"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    original_filename = Column(String(255))
    filename = Column(String(255))
    image_path = Column(String(500))
    prediction = Column(String(50))  # 'sehat' or 'sakit'
    confidence = Column(Float)
    features = Column(Text)  # JSON string of features
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class DiagnosisHistory(Base):
    """Model untuk menyimpan hasil diagnosis dari expert system"""
    __tablename__ = 'diagnosis_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=True)  # FK ke predictions table
    diagnosis = Column(Text)  # JSON string of diagnosis details
    severity = Column(String(50))  # 'ringan', 'sedang', 'berat'
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


def get_engine():
    """Return SQLAlchemy engine"""
    return engine


def init_mysql_tables():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(engine)
        print("Database tables initialized successfully")
    except SQLAlchemyError as e:
        print(f"Error initializing database tables: {e}")
        raise


def save_prediction_mysql(original_filename, filename, image_path, prediction, confidence, features_dict, timestamp=None):
    """
    Save prediction result to MySQL database
    
    Args:
        original_filename: Original filename uploaded
        filename: Saved filename
        image_path: Full path to saved image
        prediction: 'sehat' or 'sakit'
        confidence: Confidence score (0-1)
        features_dict: Dictionary of extracted features
        timestamp: Optional datetime, defaults to now
    
    Returns:
        ID of saved prediction
    """
    try:
        session = Session()
        
        # Convert features dict to JSON string
        features_json = json.dumps(features_dict, default=str)
        
        # Create prediction record
        pred = Prediction(
            original_filename=original_filename,
            filename=filename,
            image_path=image_path,
            prediction=prediction.lower(),
            confidence=float(confidence),
            features=features_json,
            timestamp=timestamp or datetime.datetime.utcnow()
        )
        
        session.add(pred)
        session.commit()
        pred_id = pred.id
        session.close()
        
        return pred_id
    except SQLAlchemyError as e:
        print(f"Error saving prediction to database: {e}")
        raise


def get_recent_predictions_mysql(limit=10):
    """
    Get recent predictions from database
    
    Args:
        limit: Number of recent predictions to retrieve
    
    Returns:
        List of dictionaries containing prediction data
    """
    try:
        session = Session()
        
        # Query recent predictions, ordered by timestamp descending
        predictions = session.query(Prediction).order_by(
            Prediction.timestamp.desc()
        ).limit(limit).all()
        
        result = []
        for pred in predictions:
            try:
                features = json.loads(pred.features) if pred.features else {}
            except:
                features = {}
            
            result.append({
                'id': pred.id,
                'original_filename': pred.original_filename,
                'filename': pred.filename,
                'image_path': pred.image_path,
                'prediction': pred.prediction,
                'confidence': round(float(pred.confidence), 4),
                'features': features,
                'timestamp': pred.timestamp.isoformat() if pred.timestamp else None
            })
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving predictions from database: {e}")
        return []


def get_prediction_by_id(pred_id):
    """
    Get specific prediction by ID
    
    Args:
        pred_id: Prediction ID
    
    Returns:
        Dictionary with prediction details or None
    """
    try:
        session = Session()
        
        pred = session.query(Prediction).filter(Prediction.id == pred_id).first()
        
        if not pred:
            session.close()
            return None
        
        try:
            features = json.loads(pred.features) if pred.features else {}
        except:
            features = {}
        
        result = {
            'id': pred.id,
            'original_filename': pred.original_filename,
            'filename': pred.filename,
            'image_path': pred.image_path,
            'prediction': pred.prediction,
            'confidence': round(float(pred.confidence), 4),
            'features': features,
            'timestamp': pred.timestamp.isoformat() if pred.timestamp else None
        }
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving prediction by ID: {e}")
        return None


def save_diagnosis_mysql(prediction_id, diagnosis_dict, severity='sedang', confidence=None, timestamp=None):
    """
    Save diagnosis result from expert system to database
    
    Args:
        prediction_id: Foreign Key ke predictions table (prediction yang di-diagnosa)
        diagnosis_dict: Dictionary of diagnosis details from expert system
        severity: 'ringan', 'sedang', or 'berat'
        confidence: Confidence score (optional)
        timestamp: Optional datetime, defaults to now
    
    Returns:
        ID of saved diagnosis
    """
    try:
        session = Session()
        
        # Convert diagnosis dict to JSON string
        diagnosis_json = json.dumps(diagnosis_dict, default=str)
        
        # Create diagnosis record with FK to predictions
        diag = DiagnosisHistory(
            prediction_id=prediction_id,
            diagnosis=diagnosis_json,
            severity=severity,
            confidence=float(confidence) if confidence else None,
            timestamp=timestamp or datetime.datetime.utcnow()
        )
        
        session.add(diag)
        session.commit()
        diag_id = diag.id
        session.close()
        
        return diag_id
    except SQLAlchemyError as e:
        print(f"Error saving diagnosis to database: {e}")
        raise


def get_diagnosis_history_mysql(limit=50, order_by='timestamp'):
    """
    Get diagnosis history from database
    
    Args:
        limit: Number of records to retrieve
        order_by: Column to order by
    
    Returns:
        List of dictionaries containing diagnosis data
    """
    try:
        session = Session()
        
        # Query diagnosis history, ordered by timestamp descending
        diagnoses = session.query(DiagnosisHistory).order_by(
            DiagnosisHistory.timestamp.desc()
        ).limit(limit).all()
        
        result = []
        for diag in diagnoses:
            try:
                diagnosis = json.loads(diag.diagnosis) if diag.diagnosis else {}
            except:
                diagnosis = {}
            
            result.append({
                'id': diag.id,
                'prediction_id': diag.prediction_id,
                'diagnosis': diagnosis,
                'severity': diag.severity,
                'confidence': round(float(diag.confidence), 4) if diag.confidence else None,
                'timestamp': diag.timestamp.isoformat() if diag.timestamp else None
            })
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving diagnosis history from database: {e}")
        return []


def get_diagnosis_by_id(diag_id):
    """
    Get specific diagnosis by ID
    
    Args:
        diag_id: Diagnosis ID
    
    Returns:
        Dictionary with diagnosis details or None
    """
    try:
        session = Session()
        
        diag = session.query(DiagnosisHistory).filter(DiagnosisHistory.id == diag_id).first()
        
        if not diag:
            session.close()
            return None
        
        try:
            diagnosis = json.loads(diag.diagnosis) if diag.diagnosis else {}
        except:
            diagnosis = {}
        
        result = {
            'id': diag.id,
            'prediction_id': diag.prediction_id,
            'diagnosis': diagnosis,
            'severity': diag.severity,
            'confidence': round(float(diag.confidence), 4) if diag.confidence else None,
            'timestamp': diag.timestamp.isoformat() if diag.timestamp else None
        }
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving diagnosis by ID: {e}")
        return None


def get_diagnosis_by_prediction_id(pred_id):
    """
    Get diagnosis for a specific prediction (by prediction_id FK)
    
    Args:
        pred_id: Prediction ID (Foreign Key)
    
    Returns:
        Dictionary with most recent diagnosis or None if no diagnosis exists
    """
    try:
        session = Session()
        
        # Get most recent diagnosis for this prediction
        diag = session.query(DiagnosisHistory).filter(
            DiagnosisHistory.prediction_id == pred_id
        ).order_by(DiagnosisHistory.timestamp.desc()).first()
        
        if not diag:
            session.close()
            return None
        
        try:
            diagnosis = json.loads(diag.diagnosis) if diag.diagnosis else {}
        except:
            diagnosis = {}
        
        result = {
            'id': diag.id,
            'prediction_id': diag.prediction_id,
            'diagnosis': diagnosis,
            'severity': diag.severity,
            'confidence': round(float(diag.confidence), 4) if diag.confidence else None,
            'timestamp': diag.timestamp.isoformat() if diag.timestamp else None
        }
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving diagnosis by prediction ID: {e}")
        return None


def delete_prediction_mysql(pred_id):
    """Delete a prediction by ID"""
    try:
        session = Session()
        session.query(Prediction).filter(Prediction.id == pred_id).delete()
        session.commit()
        session.close()
        return True
    except SQLAlchemyError as e:
        print(f"Error deleting prediction: {e}")
        return False


def delete_diagnosis_mysql(diag_id):
    """Delete a diagnosis by ID"""
    try:
        session = Session()
        session.query(DiagnosisHistory).filter(DiagnosisHistory.id == diag_id).delete()
        session.commit()
        session.close()
        return True
    except SQLAlchemyError as e:
        print(f"Error deleting diagnosis: {e}")
        return False


def get_statistics_mysql():
    """Get statistics from database"""
    try:
        session = Session()
        
        # Count predictions
        total_predictions = session.query(Prediction).count()
        sakit_predictions = session.query(Prediction).filter(Prediction.prediction == 'sakit').count()
        sehat_predictions = session.query(Prediction).filter(Prediction.prediction == 'sehat').count()
        
        # Count diagnoses
        total_diagnoses = session.query(DiagnosisHistory).count()
        
        # Average confidence
        avg_confidence = None
        result = session.query(Prediction).first()
        if result:
            from sqlalchemy import func
            avg_conf = session.query(func.avg(Prediction.confidence)).scalar()
            avg_confidence = round(float(avg_conf), 4) if avg_conf else None
        
        session.close()
        
        return {
            'total_predictions': total_predictions,
            'sakit_predictions': sakit_predictions,
            'sehat_predictions': sehat_predictions,
            'total_diagnoses': total_diagnoses,
            'average_confidence': avg_confidence
        }
    except SQLAlchemyError as e:
        print(f"Error getting statistics: {e}")
        return {}
