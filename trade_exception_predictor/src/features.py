"""
features.py - Feature Engineering for Trade Reconciliation Dataset

Works with these exact columns:
  trade_id, trade_date, settlement_date, days_to_settlement, commodity_type,
  instrument_type, delivery_location, counterparty_id, counterparty_tier,
  counterparty_region, notional_usd, quantity_mt, price_per_mt,
  settlement_currency, is_month_end, is_quarter_end, day_of_week,
  counterparty_exception_rate_30d, same_commodity_breaks_7d, price_volatility_flag,
  cross_border_flag, currency_mismatch_flag, documentation_lag_days,
  amendment_count, is_exception
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineering:
    """Feature engineering pipeline for trade reconciliation data."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def engineer_features(self, test_size=0.2, random_state=42):
        """
        Full feature engineering pipeline.
        
        Returns
        -------
        X_train, X_test, y_train, y_test, feature_names
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*80)
        
        # Step 1: Parse dates
        print("\n1. Parsing dates...")
        self._parse_dates()
        
        # Step 2: Encode categorical features
        print("2. Encoding categorical features...")
        self._encode_categorical()
        
        # Step 3: Create interaction features
        print("3. Creating interaction features...")
        self._create_interactions()
        
        # Step 4: Scale numeric features
        print("4. Scaling numeric features...")
        self._scale_numeric()
        
        # Step 5: Remove non-feature columns
        print("5. Cleaning non-feature columns...")
        self._cleanup()
        
        # Step 6: Prepare train/test split
        print("6. Splitting data...")
        y = self.df['is_exception'].values
        X = self.df.drop('is_exception', axis=1).values
        feature_names = self.df.drop('is_exception', axis=1).columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        self.is_fitted = True
        self.feature_names = feature_names
        
        # Summary
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        print("\n" + "="*80)
        print("✓ FEATURE ENGINEERING COMPLETE")
        print("="*80)
        print(f"\nDataset Summary:")
        print(f"  • Features: {len(feature_names)}")
        print(f"  • Train samples: {len(X_train):,}")
        print(f"  • Test samples: {len(X_test):,}")
        print(f"  • Exception rate (train): {y_train.mean():.1%}")
        print(f"  • Exception rate (test): {y_test.mean():.1%}")
        print(f"  • scale_pos_weight: {scale_pos_weight:.2f}")
        print()
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def _parse_dates(self):
        """Convert date columns to datetime."""
        if 'trade_date' in self.df.columns:
            self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        if 'settlement_date' in self.df.columns:
            self.df['settlement_date'] = pd.to_datetime(self.df['settlement_date'])
    
    def _encode_categorical(self):
        """
        Encode categorical features.
        
        One-Hot: commodity_type, instrument_type, delivery_location, 
                 settlement_currency, day_of_week
        Label: counterparty_id, counterparty_tier, counterparty_region
        """
        # One-Hot encode
        one_hot_cols = [
            'commodity_type', 'instrument_type', 'delivery_location',
            'settlement_currency', 'day_of_week'
        ]
        
        for col in one_hot_cols:
            if col in self.df.columns:
                encoded = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, encoded], axis=1)
                self.df.drop(col, axis=1, inplace=True)
        
        # Label encode
        label_cols = ['counterparty_id', 'counterparty_tier', 'counterparty_region']
        
        for col in label_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
    
    def _create_interactions(self):
        """Create interaction features based on domain knowledge."""
        
        # Tier × Month-End: small traders struggle more under stress
        if 'counterparty_tier' in self.df.columns and 'is_month_end' in self.df.columns:
            self.df['tier_x_month_end'] = (
                self.df['counterparty_tier'] * self.df['is_month_end']
            )
        
        # Tier × Doc Lag: late docs from risky parties is worse
        if 'counterparty_tier' in self.df.columns and 'documentation_lag_days' in self.df.columns:
            self.df['tier_x_doc_lag'] = (
                self.df['counterparty_tier'] * self.df['documentation_lag_days']
            )
        
        # High doc lag flag
        if 'documentation_lag_days' in self.df.columns:
            self.df['high_doc_lag'] = (self.df['documentation_lag_days'] > 3).astype(int)
        
        # High amendments flag
        if 'amendment_count' in self.df.columns:
            self.df['high_amendments'] = (self.df['amendment_count'] > 2).astype(int)
        
        # Doc lag × Amendment interaction
        if 'documentation_lag_days' in self.df.columns and 'amendment_count' in self.df.columns:
            self.df['doc_lag_x_amendments'] = (
                self.df['documentation_lag_days'] * self.df['amendment_count']
            )
    
    def _scale_numeric(self):
        """Scale numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'is_exception']
        
        if numeric_cols:
            self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
    
    def _cleanup(self):
        """Remove non-feature columns."""
        drop_cols = ['trade_id', 'trade_date', 'settlement_date']
        self.df = self.df.drop([col for col in drop_cols if col in self.df.columns], axis=1)
        
        # Keep only numeric
        self.df = self.df.select_dtypes(include=[np.number])
    
    def transform(self, df_new):
        """Apply same transformations to new data."""
        if not self.is_fitted:
            raise ValueError("Must fit first with engineer_features()")
        
        df = df_new.copy()
        
        # Parse dates
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        if 'settlement_date' in df.columns:
            df['settlement_date'] = pd.to_datetime(df['settlement_date'])
        
        # One-Hot encode same columns
        one_hot_cols = [
            'commodity_type', 'instrument_type', 'delivery_location',
            'settlement_currency', 'day_of_week'
        ]
        
        for col in one_hot_cols:
            if col in df.columns:
                encoded = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, encoded], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        # Label encode same columns (handle unseen labels)
        for col, le in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    # Handle unseen labels by mapping to the first known class
                    mask = df[col].isin(le.classes_)
                    df.loc[mask, col] = le.transform(df.loc[mask, col])
                    df.loc[~mask, col] = le.transform(le.classes_[[0]])[0]
        
        # Create interactions
        if 'counterparty_tier' in df.columns and 'is_month_end' in df.columns:
            df['tier_x_month_end'] = df['counterparty_tier'] * df['is_month_end']
        
        if 'counterparty_tier' in df.columns and 'documentation_lag_days' in df.columns:
            df['tier_x_doc_lag'] = df['counterparty_tier'] * df['documentation_lag_days']
        
        if 'documentation_lag_days' in df.columns:
            df['high_doc_lag'] = (df['documentation_lag_days'] > 3).astype(int)
        
        if 'amendment_count' in df.columns:
            df['high_amendments'] = (df['amendment_count'] > 2).astype(int)
        
        if 'documentation_lag_days' in df.columns and 'amendment_count' in df.columns:
            df['doc_lag_x_amendments'] = df['documentation_lag_days'] * df['amendment_count']
        
        # Scale
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # Cleanup
        df = df.drop([col for col in ['trade_id', 'trade_date', 'settlement_date'] if col in df.columns], axis=1)
        df = df.select_dtypes(include=[np.number])
        
        return df.values
    
    def get_feature_names(self):
        """Return final feature names."""
        if not self.is_fitted:
            raise ValueError("Must fit first")
        return self.feature_names


def prepare_features(df, test_size=0.2, random_state=42):
    """Quick interface to feature engineering pipeline."""
    fe = FeatureEngineering(df)
    return fe.engineer_features(test_size=test_size, random_state=random_state)