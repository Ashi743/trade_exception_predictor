"""Feature engineering for trade exception prediction."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple


class FeatureEngineering:
    """Engineer features from raw trade data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.encoders = {}

    def engineer_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features and prepare target."""
        X = self.df.drop(['trade_id', 'timestamp', 'settlement_date', 'is_exception'], axis=1)
        y = self.df['is_exception']

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le

        # Create interaction features
        X['volatility_risk'] = X['market_volatility'] * X['counterparty_risk_score']
        X['speed_deviation'] = X['execution_speed_ms'] * X['price_deviation_pct']
        X['size_volatility'] = X['trade_size_percentile'] * X['market_volatility']

        # Scale features
        X = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        return X, y

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply same transformations to new data."""
        # Encode categorical features
        for col, le in self.encoders.items():
            if col in X.columns:
                X[col] = le.transform(X[col])

        # Create interaction features if columns exist
        if 'market_volatility' in X.columns and 'counterparty_risk_score' in X.columns:
            X['volatility_risk'] = X['market_volatility'] * X['counterparty_risk_score']
        if 'execution_speed_ms' in X.columns and 'price_deviation_pct' in X.columns:
            X['speed_deviation'] = X['execution_speed_ms'] * X['price_deviation_pct']
        if 'trade_size_percentile' in X.columns and 'market_volatility' in X.columns:
            X['size_volatility'] = X['trade_size_percentile'] * X['market_volatility']

        # Scale
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
