import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_model(self, model_path='models/model.pkl', scaler_path='models/scaler.pkl'):
        """Load the trained model and scaler"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, features):
        """Make predictions using the loaded model"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded")
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(scaled_features)
        probability = self.model.predict_proba(scaled_features)
        
        return {
            'prediction': prediction[0],
            'probability': probability[0].max(),
            'class_probabilities': dict(zip(self.model.classes_, probability[0]))
        }