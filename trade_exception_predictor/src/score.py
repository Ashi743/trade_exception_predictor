"""
score.py - Azure ML Inference Service with SHAP Explanations

Works with exact trade columns:
  trade_id, trade_date, settlement_date, days_to_settlement, commodity_type,
  instrument_type, delivery_location, counterparty_id, counterparty_tier,
  counterparty_region, notional_usd, quantity_mt, price_per_mt,
  settlement_currency, is_month_end, is_quarter_end, day_of_week,
  counterparty_exception_rate_30d, same_commodity_breaks_7d, price_volatility_flag,
  cross_border_flag, currency_mismatch_flag, documentation_lag_days,
  amendment_count, is_exception
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')


class TradeExceptionScoringService:
    """Scoring service for trade exception predictions with SHAP explanations."""
    
    def __init__(self, model_path, feature_engineer_path, feature_names):
        """
        Initialize scoring service.
        
        Parameters
        ----------
        model_path : str
            Path to saved XGBoost model
        feature_engineer_path : str
            Path to saved FeatureEngineering object
        feature_names : list
            List of final feature names (after engineering)
        """
        self.model = joblib.load(model_path)
        self.feature_engineer = joblib.load(feature_engineer_path)
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)
        
    def score_trade(self, trade_data):
        """
        Score a single trade with SHAP explanation.
        
        Parameters
        ----------
        trade_data : dict
            Trade features as dictionary
            {
                'trade_id': 'TR_001',
                'trade_date': '2024-01-15',
                'settlement_date': '2024-01-20',
                'days_to_settlement': 5,
                'commodity_type': 'Soybean',
                'counterparty_tier': 'Tier1',
                'documentation_lag_days': 2,
                ...
            }
        
        Returns
        -------
        dict
            {
                'status': 'success',
                'trade_id': 'TR_001',
                'prediction': 0,  # 0=clean, 1=exception
                'prediction_label': 'CLEAN',
                'exception_probability': 0.1234,
                'confidence': 'high',
                'top_drivers': [...],
                'recommendation': 'Clean trade...'
            }
        """
        
        try:
            # Extract trade_id for reference
            trade_id = trade_data.get('trade_id', 'unknown')
            
            # Convert to DataFrame
            df = pd.DataFrame([trade_data])
            
            # Apply feature engineering
            X = self.feature_engineer.transform(df)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            exception_prob = float(probabilities[1])
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle binary classification output
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # Exception class
            else:
                shap_vals = shap_values[0]
            
            # Get top drivers
            top_drivers = self._get_top_drivers(shap_vals, top_k=5)
            
            # Determine confidence
            if exception_prob < 0.3:
                confidence = 'high'
                status_emoji = '✓'
            elif exception_prob < 0.7:
                confidence = 'medium'
                status_emoji = '⚠️'
            else:
                confidence = 'high'
                status_emoji = '🚨'
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                prediction,
                exception_prob,
                top_drivers,
                status_emoji
            )
            
            response = {
                'status': 'success',
                'trade_id': trade_id,
                'prediction': int(prediction),
                'prediction_label': 'EXCEPTION' if prediction == 1 else 'CLEAN',
                'exception_probability': round(exception_prob, 4),
                'clean_probability': round(float(probabilities[0]), 4),
                'confidence': confidence,
                'top_drivers': top_drivers,
                'recommendation': recommendation,
                'base_rate': round(float(self.explainer.expected_value), 4)
            }
            
            return response
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'trade_id': trade_data.get('trade_id', 'unknown')
            }
    
    def _get_top_drivers(self, shap_vals, top_k=5):
        """
        Extract top SHAP drivers for a prediction.
        
        Returns list of dicts with feature, shap_value, direction
        """
        contributions = []
        
        for feature_name, shap_val in zip(self.feature_names, shap_vals):
            contributions.append({
                'feature': feature_name,
                'shap_value': float(shap_val),
                'direction': 'increases_exception_risk' if shap_val > 0 else 'decreases_exception_risk',
                'magnitude': abs(float(shap_val))
            })
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: x['magnitude'], reverse=True)
        
        # Return top K without magnitude field
        return [
            {k: v for k, v in item.items() if k != 'magnitude'}
            for item in contributions[:top_k]
        ]
    
    def _generate_recommendation(self, prediction, prob, drivers, emoji):
        """Generate operational recommendation based on prediction."""
        
        if prediction == 1:  # Exception predicted
            if prob > 0.9:
                return (
                    f"{emoji} CRITICAL RISK: {100*prob:.0f}% exception probability. "
                    f"Recommend immediate review and escalation. "
                    f"Primary issue: {drivers[0]['feature']}"
                )
            elif prob > 0.7:
                return (
                    f"{emoji} HIGH RISK: {100*prob:.0f}% exception probability. "
                    f"Recommend follow-up review within 24 hours. "
                    f"Top driver: {drivers[0]['feature']}"
                )
            else:
                return (
                    f"{emoji} MEDIUM RISK: {100*prob:.0f}% exception probability. "
                    f"Monitor closely, may need documentation or counterparty follow-up."
                )
        else:  # Clean predicted
            if prob < 0.1:
                return f"{emoji} VERY LOW RISK: {100*prob:.0f}% exception probability. Trade is clean."
            elif prob < 0.2:
                return f"{emoji} LOW RISK: {100*prob:.0f}% exception probability. Standard processing recommended."
            else:
                return f"{emoji} ACCEPTABLE: {100*prob:.0f}% exception probability. Routine settlement expected."


# ============================================================================
# AZURE ML ENDPOINT INTERFACE
# ============================================================================

# Global variables for Azure ML endpoint initialization
_model = None
_feature_engineer = None
_service = None
_feature_names = None


def init():
    """
    Called once when Azure ML endpoint initializes.
    Loads model and feature engineering objects.
    """
    global _model, _feature_engineer, _service, _feature_names
    
    try:
        print("Initializing Trade Exception Scoring Service...")
        
        # Standard Azure ML output paths
        model_path = 'outputs/xgboost_model.pkl'
        feature_engineer_path = 'outputs/feature_engineer.pkl'
        
        # Feature names from trained model
        _feature_names = [
            'days_to_settlement', 'notional_usd', 'quantity_mt', 'price_per_mt',
            'counterparty_exception_rate_30d', 'same_commodity_breaks_7d',
            'is_month_end', 'is_quarter_end', 'price_volatility_flag',
            'cross_border_flag', 'currency_mismatch_flag',
            'documentation_lag_days', 'amendment_count',
            'counterparty_id', 'counterparty_tier', 'counterparty_region',
            # One-hot encoded columns (examples - actual will depend on your data)
            'commodity_type_Corn', 'commodity_type_Soybean', 'commodity_type_Wheat',
            'instrument_type_Forward', 'instrument_type_Futures', 'instrument_type_Spot',
            'settlement_currency_BRL', 'settlement_currency_CNY', 'settlement_currency_EUR',
            'day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday',
            # Interaction features
            'tier_x_month_end', 'tier_x_doc_lag', 'high_doc_lag',
            'high_amendments', 'doc_lag_x_amendments'
        ]
        
        _service = TradeExceptionScoringService(
            model_path,
            feature_engineer_path,
            _feature_names
        )
        
        print("✓ Scoring service initialized successfully")
    
    except Exception as e:
        print(f"✗ Error initializing service: {str(e)}")
        raise


def run(raw_data):
    """
    Called for each prediction request.
    
    Input (JSON):
    {
        "trade_id": "TR_001",
        "trade_date": "2024-01-15",
        "settlement_date": "2024-01-20",
        "days_to_settlement": 5,
        "commodity_type": "Soybean",
        "instrument_type": "Spot",
        "delivery_location": "Port_A",
        "counterparty_id": "CP_001",
        "counterparty_tier": "Tier1",
        "counterparty_region": "North_America",
        "notional_usd": 100000,
        "quantity_mt": 100,
        "price_per_mt": 500,
        "settlement_currency": "USD",
        "is_month_end": 0,
        "is_quarter_end": 0,
        "day_of_week": "Monday",
        "counterparty_exception_rate_30d": 0.25,
        "same_commodity_breaks_7d": 1,
        "price_volatility_flag": 0,
        "cross_border_flag": 0,
        "currency_mismatch_flag": 0,
        "documentation_lag_days": 2,
        "amendment_count": 0
    }
    
    Output (JSON):
    {
        "status": "success",
        "trade_id": "TR_001",
        "prediction": 0,
        "prediction_label": "CLEAN",
        "exception_probability": 0.1234,
        "clean_probability": 0.8766,
        "confidence": "high",
        "top_drivers": [
            {
                "feature": "documentation_lag_days",
                "shap_value": -0.0521,
                "direction": "decreases_exception_risk"
            },
            ...
        ],
        "recommendation": "✓ VERY LOW RISK: Trade is clean.",
        "base_rate": 0.2910
    }
    """
    
    try:
        # Parse input
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data
        
        # Score
        response = _service.score_trade(data)
        
        return json.dumps(response)
    
    except Exception as e:
        error_response = {
            'status': 'error',
            'error': f'Scoring failed: {str(e)}',
            'trade_id': data.get('trade_id', 'unknown') if 'data' in locals() else 'unknown'
        }
        return json.dumps(error_response)


# ============================================================================
# BATCH SCORING INTERFACE
# ============================================================================

def batch_score(data_df):
    """
    Score multiple trades in batch.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame with one row per trade (all 25 columns)
    
    Returns
    -------
    pd.DataFrame
        Original data plus prediction columns
    """
    
    results = []
    
    for idx, row in data_df.iterrows():
        trade_data = row.to_dict()
        response = _service.score_trade(trade_data)
        results.append(response)
    
    results_df = pd.DataFrame(results)
    return pd.concat([data_df, results_df], axis=1)


# ============================================================================
# LOCAL TESTING
# ============================================================================

def test_scoring():
    """Test scoring service locally."""
    
    # Example trade
    test_trade = {
        'trade_id': 'TR_TEST_001',
        'trade_date': '2024-01-15',
        'settlement_date': '2024-01-20',
        'days_to_settlement': 5,
        'commodity_type': 'Soybean',
        'instrument_type': 'Spot',
        'delivery_location': 'Port_A',
        'counterparty_id': 'CP_001',
        'counterparty_tier': 'Tier1',
        'counterparty_region': 'North_America',
        'notional_usd': 100000.0,
        'quantity_mt': 100.0,
        'price_per_mt': 500.0,
        'settlement_currency': 'USD',
        'is_month_end': 0,
        'is_quarter_end': 0,
        'day_of_week': 'Monday',
        'counterparty_exception_rate_30d': 0.25,
        'same_commodity_breaks_7d': 1,
        'price_volatility_flag': 0,
        'cross_border_flag': 0,
        'currency_mismatch_flag': 0,
        'documentation_lag_days': 2,
        'amendment_count': 0
    }
    
    result = _service.score_trade(test_trade)
    
    print("\n" + "="*80)
    print("TRADE EXCEPTION PREDICTOR - SCORING TEST")
    print("="*80)
    print(f"\nTrade ID: {result['trade_id']}")
    print(f"Prediction: {result['prediction_label']}")
    print(f"Exception Probability: {result['exception_probability']:.1%}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nTop Risk Drivers:")
    for i, driver in enumerate(result['top_drivers'][:5], 1):
        print(f"  {i}. {driver['feature']:40s} | SHAP: {driver['shap_value']:+.4f} ({driver['direction']})")
    print(f"\nRecommendation: {result['recommendation']}")
    print("\n" + "="*80)
    
    return result


if __name__ == '__main__':
    # Initialize
    init()
    
    # Test
    test_scoring()