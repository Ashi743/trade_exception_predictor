"""
explain.py - SHAP Explainability Module

Provides interpretable explanations for trade exception predictions.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ExplainabilityAnalyzer:
    """SHAP-based explainability for trade predictions."""
    
    def __init__(self, model, X_test, y_test=None):
        """Initialize explainer."""
        self.model = model
        self.X_test = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        self.y_test = y_test
        
        print("\nComputing SHAP values...")
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(self.X_test)
        
        # Handle binary classification
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        print("✓ SHAP values computed\n")
    
    def get_feature_importance(self, top_k=15):
        """Get feature importance ranking."""
        feature_importance = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': [str(f) for f in self.X_test.columns],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        print("\n" + "─"*70)
        print("FEATURE IMPORTANCE (by SHAP)")
        print("─"*70)

        for idx, row in importance_df.head(top_k).iterrows():
            pct = 100 * row['importance'] / importance_df['importance'].sum()
            bar_len = int(pct / 2)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            print(f"{row['feature']:35s} │ {bar} {pct:5.1f}%")

        print()
        return importance_df
    
    def explain_prediction(self, instance_idx, top_k=10):
        """Explain a single prediction."""
        shap_vals = self.shap_values[instance_idx]

        explanation_df = pd.DataFrame({
            'feature': [str(f) for f in self.X_test.columns],
            'value': self.X_test.iloc[instance_idx].values,
            'shap_value': shap_vals
        }).sort_values('shap_value', key=abs, ascending=False)
        
        # Prediction details
        pred_proba = self.model.predict_proba(self.X_test.iloc[[instance_idx]])[0]
        
        details = {
            'instance_idx': instance_idx,
            'exception_probability': float(pred_proba[1]),
            'clean_probability': float(pred_proba[0]),
            'base_value': float(self.explainer.expected_value)
        }
        
        print("\n" + "─"*70)
        print(f"PREDICTION EXPLANATION - INSTANCE #{instance_idx}")
        print("─"*70)
        print(f"Exception probability: {details['exception_probability']:.1%}")
        print(f"Clean probability:     {details['clean_probability']:.1%}")
        print(f"Base rate:             {details['base_value']:.1%}")
        
        print(f"\nTop {top_k} Contributing Factors:")
        print("─"*70)
        
        for idx, row in explanation_df.head(top_k).iterrows():
            direction = "↑ increases risk" if row['shap_value'] > 0 else "↓ decreases risk"
            print(f"{row['feature']:35s} │ {direction:20s}")
            print(f"  Value: {row['value']:8.4f}, SHAP: {row['shap_value']:+.4f}")
        
        print()
        return explanation_df.head(top_k), details
    
    def plot_summary(self, max_display=15):
        """Plot SHAP summary."""
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        return fig
    
    def plot_dependence(self, feature_name):
        """Plot SHAP dependence for a feature."""
        if feature_name not in self.X_test.columns:
            raise ValueError(f"Feature '{feature_name}' not found")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            self.X_test,
            show=False
        )
        plt.tight_layout()
        return fig
    
    def plot_waterfall(self, instance_idx):
        """Plot SHAP waterfall for single prediction."""
        explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=self.explainer.expected_value,
            data=self.X_test.iloc[instance_idx],
            feature_names=[str(f) for f in self.X_test.columns]
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        return fig
    
    def get_high_risk_trades(self, top_n=10):
        """Get explanations for highest-risk trades."""
        pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        top_indices = np.argsort(-pred_proba)[:top_n]
        
        print("\n" + "─"*70)
        print(f"TOP {top_n} HIGHEST-RISK TRADES")
        print("─"*70)
        
        explanations = []
        for rank, idx in enumerate(top_indices, 1):
            exp_df, details = self.explain_prediction(idx, top_k=3)
            explanations.append({
                'rank': rank,
                'instance_idx': idx,
                'exception_probability': details['exception_probability'],
                'top_drivers': exp_df.to_dict('records')
            })
            
            print(f"\n{rank}. Instance #{idx} - {details['exception_probability']:.1%} risk")
            for _, driver in exp_df.head(3).iterrows():
                print(f"   • {driver['feature']}: {driver['shap_value']:+.4f}")
        
        return explanations
    
    def compare_predictions(self, instance_indices):
        """Compare multiple predictions."""
        comparison_data = []
        
        for idx in instance_indices:
            exp_df, details = self.explain_prediction(idx, top_k=1)
            comparison_data.append({
                'instance': idx,
                'prediction': 'EXCEPTION' if details['exception_probability'] > 0.5 else 'CLEAN',
                'probability': details['exception_probability'],
                'top_feature': exp_df.iloc[0]['feature'] if len(exp_df) > 0 else 'N/A'
            })
        
        return pd.DataFrame(comparison_data)