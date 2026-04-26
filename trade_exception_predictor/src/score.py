"""Inference script for Azure ML endpoint."""
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from features import FeatureEngineering


class TradeExceptionScoringService:
    """Scoring service for Azure ML endpoint."""

    def __init__(self, model_path: str, encoder_path: str):
        self.model = joblib.load(model_path)
        self.feature_engineer = joblib.load(encoder_path)
        self.feature_names = None

    def init(self):
        """Called once when service initializes."""
        pass

    def run(self, raw_data):
        """Score incoming data."""
        try:
            # Parse input
            if isinstance(raw_data, str):
                data = json.loads(raw_data)
            else:
                data = raw_data

            # Convert to DataFrame
            df = pd.DataFrame([data])

            # Feature engineering
            X = self.feature_engineer.transform(df)

            # Predictions
            y_pred = self.model.predict(X)[0]
            y_proba = self.model.predict_proba(X)[0]

            # Format response
            response = {
                'prediction': int(y_pred),
                'probability_normal': float(y_proba[0]),
                'probability_exception': float(y_proba[1]),
                'status': 'success'
            }

            return json.dumps(response)

        except Exception as e:
            return json.dumps({
                'error': str(e),
                'status': 'error'
            })


def init():
    """Initialize model for batch scoring."""
    global model, feature_engineer

    model = joblib.load('model.pkl')
    feature_engineer = joblib.load('feature_engineer.pkl')


def run(raw_data):
    """Run batch scoring."""
    data = json.loads(raw_data)
    df = pd.DataFrame([data])

    X = feature_engineer.transform(df)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return json.dumps({
        'prediction': int(predictions[0]),
        'probability_normal': float(probabilities[0][0]),
        'probability_exception': float(probabilities[0][1])
    })
