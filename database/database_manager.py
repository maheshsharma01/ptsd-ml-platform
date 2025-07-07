import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import streamlit as st

Base = declarative_base()

class PatientData(Base):
    __tablename__ = 'patient_data'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50), unique=True, nullable=False)
    age = Column(Integer)
    gender = Column(String(10))
    trauma_type = Column(String(100))
    trauma_severity = Column(Float)
    pcl5_intrusive = Column(Float)
    pcl5_avoidance = Column(Float)
    pcl5_negative_mood = Column(Float)
    pcl5_hyperarousal = Column(Float)
    pcl5_total_score = Column(Float)
    cortisol_level = Column(Float)
    time_since_trauma = Column(Integer)
    social_support_score = Column(Float)
    education_years = Column(Integer)
    employment_status = Column(Integer)
    previous_trauma = Column(Boolean)
    family_history_mental_health = Column(Boolean)
    ptsd_diagnosis = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelResults(Base):
    __tablename__ = 'model_results'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    cv_mean = Column(Float)
    cv_std = Column(Float)
    confusion_matrix = Column(Text)  # JSON string
    hyperparameters = Column(Text)  # JSON string
    training_data_size = Column(Integer)
    feature_count = Column(Integer)
    training_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    prediction = Column(Integer)  # 0 or 1
    probability = Column(Float)
    feature_values = Column(Text)  # JSON string
    prediction_date = Column(DateTime, default=datetime.utcnow)
    risk_level = Column(String(20))
    notes = Column(Text)

class DatabaseManager:
    """
    Database manager for PTSD ML platform.
    Handles patient data, model results, and prediction history.
    """
    
    def __init__(self):
        self.engine = None
        self.Session = None
        self.connect()
    
    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                st.error("Database connection string not found. Please check environment variables.")
                return False
            
            self.engine = create_engine(database_url)
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            return True
            
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return False
    
    def save_patient_data(self, patient_data_dict):
        """
        Save patient data to database.
        
        Args:
            patient_data_dict: Dictionary with patient information
            
        Returns:
            Boolean indicating success
        """
        try:
            session = self.Session()
            
            # Check if patient exists
            existing_patient = session.query(PatientData).filter_by(
                patient_id=patient_data_dict.get('patient_id')
            ).first()
            
            if existing_patient:
                # Update existing patient
                for key, value in patient_data_dict.items():
                    if hasattr(existing_patient, key):
                        setattr(existing_patient, key, value)
            else:
                # Create new patient record
                patient = PatientData(**patient_data_dict)
                session.add(patient)
            
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            st.error(f"Error saving patient data: {str(e)}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def save_model_results(self, model_name, results_dict):
        """
        Save model training results to database.
        
        Args:
            model_name: Name of the ML model
            results_dict: Dictionary with model performance metrics
            
        Returns:
            Boolean indicating success
        """
        try:
            session = self.Session()
            
            # Convert confusion matrix to JSON string
            confusion_matrix_json = json.dumps(
                results_dict.get('confusion_matrix', []).tolist() 
                if isinstance(results_dict.get('confusion_matrix'), np.ndarray) 
                else results_dict.get('confusion_matrix', [])
            )
            
            model_result = ModelResults(
                model_name=model_name,
                accuracy=results_dict.get('accuracy'),
                precision_score=results_dict.get('precision'),
                recall=results_dict.get('recall'),
                f1_score=results_dict.get('f1_score'),
                auc_score=results_dict.get('auc'),
                cv_mean=results_dict.get('cv_mean'),
                cv_std=results_dict.get('cv_std'),
                confusion_matrix=confusion_matrix_json,
                training_data_size=results_dict.get('training_data_size'),
                feature_count=results_dict.get('feature_count')
            )
            
            session.add(model_result)
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            st.error(f"Error saving model results: {str(e)}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def save_prediction(self, patient_id, model_name, prediction, probability, 
                       feature_values, risk_level=None, notes=None):
        """
        Save prediction result to database.
        
        Args:
            patient_id: Patient identifier
            model_name: Name of the ML model used
            prediction: Prediction result (0 or 1)
            probability: Prediction probability
            feature_values: Dictionary of feature values used
            risk_level: Risk level classification
            notes: Additional notes
            
        Returns:
            Boolean indicating success
        """
        try:
            session = self.Session()
            
            prediction_record = PredictionHistory(
                patient_id=patient_id,
                model_name=model_name,
                prediction=prediction,
                probability=probability,
                feature_values=json.dumps(feature_values),
                risk_level=risk_level,
                notes=notes
            )
            
            session.add(prediction_record)
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            st.error(f"Error saving prediction: {str(e)}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def get_patient_data(self, patient_id=None, limit=100):
        """
        Retrieve patient data from database.
        
        Args:
            patient_id: Specific patient ID (optional)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with patient data
        """
        try:
            session = self.Session()
            
            if patient_id:
                query = session.query(PatientData).filter_by(patient_id=patient_id)
            else:
                query = session.query(PatientData).limit(limit)
            
            patients = query.all()
            session.close()
            
            # Convert to DataFrame
            if patients:
                data = []
                for patient in patients:
                    patient_dict = {
                        column.name: getattr(patient, column.name)
                        for column in patient.__table__.columns
                    }
                    data.append(patient_dict)
                
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error retrieving patient data: {str(e)}")
            return pd.DataFrame()
    
    def get_model_results(self, model_name=None):
        """
        Retrieve model results from database.
        
        Args:
            model_name: Specific model name (optional)
            
        Returns:
            DataFrame with model results
        """
        try:
            session = self.Session()
            
            if model_name:
                query = session.query(ModelResults).filter_by(model_name=model_name)
            else:
                query = session.query(ModelResults)
            
            results = query.order_by(ModelResults.training_date.desc()).all()
            session.close()
            
            # Convert to DataFrame
            if results:
                data = []
                for result in results:
                    result_dict = {
                        column.name: getattr(result, column.name)
                        for column in result.__table__.columns
                    }
                    data.append(result_dict)
                
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error retrieving model results: {str(e)}")
            return pd.DataFrame()
    
    def get_prediction_history(self, patient_id=None, model_name=None, limit=100):
        """
        Retrieve prediction history from database.
        
        Args:
            patient_id: Specific patient ID (optional)
            model_name: Specific model name (optional)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with prediction history
        """
        try:
            session = self.Session()
            
            query = session.query(PredictionHistory)
            
            if patient_id:
                query = query.filter_by(patient_id=patient_id)
            
            if model_name:
                query = query.filter_by(model_name=model_name)
            
            predictions = query.order_by(PredictionHistory.prediction_date.desc()).limit(limit).all()
            session.close()
            
            # Convert to DataFrame
            if predictions:
                data = []
                for prediction in predictions:
                    pred_dict = {
                        column.name: getattr(prediction, column.name)
                        for column in prediction.__table__.columns
                    }
                    data.append(pred_dict)
                
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error retrieving prediction history: {str(e)}")
            return pd.DataFrame()
    
    def get_database_stats(self):
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            session = self.Session()
            
            stats = {
                'total_patients': session.query(PatientData).count(),
                'total_predictions': session.query(PredictionHistory).count(),
                'total_models': session.query(ModelResults).count(),
                'ptsd_positive_cases': session.query(PatientData).filter_by(ptsd_diagnosis=1).count(),
                'ptsd_negative_cases': session.query(PatientData).filter_by(ptsd_diagnosis=0).count()
            }
            
            session.close()
            return stats
            
        except Exception as e:
            st.error(f"Error retrieving database statistics: {str(e)}")
            return {}
    
    def bulk_insert_patient_data(self, df):
        """
        Bulk insert patient data from DataFrame.
        
        Args:
            df: Pandas DataFrame with patient data
            
        Returns:
            Boolean indicating success
        """
        try:
            session = self.Session()
            
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Create PatientData objects
            patient_objects = []
            for record in records:
                # Filter only columns that exist in the PatientData model
                filtered_record = {
                    key: value for key, value in record.items()
                    if hasattr(PatientData, key) and value is not None
                }
                patient_objects.append(PatientData(**filtered_record))
            
            # Bulk insert
            session.bulk_save_objects(patient_objects)
            session.commit()
            session.close()
            
            return True
            
        except Exception as e:
            st.error(f"Error bulk inserting patient data: {str(e)}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def delete_patient_data(self, patient_id):
        """
        Delete patient data and associated predictions.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Boolean indicating success
        """
        try:
            session = self.Session()
            
            # Delete predictions first (foreign key constraint)
            session.query(PredictionHistory).filter_by(patient_id=patient_id).delete()
            
            # Delete patient data
            session.query(PatientData).filter_by(patient_id=patient_id).delete()
            
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            st.error(f"Error deleting patient data: {str(e)}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False