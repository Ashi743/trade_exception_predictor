"""Model explainability using SHAP."""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Optional


class ExplainabilityAnalyzer:
    """Analyze model predictions using SHAP."""

    def __init__(self, model, X_data: pd.DataFrame, y_data: Optional[pd.Series] = None):
        self.model = model
        self.X_data = X_data
        self.y_data = y_data
        self.explainer = None
        self.shap_values = None
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Initialize SHAP explainer."""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(self.X_data)
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get mean absolute SHAP values as importance."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        # Handle both single output and multi-output cases
        if isinstance(self.shap_values, list):
            shap_vals = np.abs(self.shap_values[1]).mean(axis=0)
        else:
            shap_vals = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.X_data.columns,
            'importance': shap_vals
        }).sort_values('importance', ascending=False)

        return importance_df

    def plot_shap_summary(self, plot_type: str = 'bar', max_display: int = 10) -> None:
        """Plot SHAP summary."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        # Handle multi-output case
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_vals,
            self.X_data,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.show()

    def plot_shap_dependence(self, feature: str, interaction_index: str = 'auto') -> None:
        """Plot SHAP dependence for a feature."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        if feature not in self.X_data.columns:
            raise ValueError(f"Feature '{feature}' not found in data")

        # Handle multi-output case
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values

        feature_idx = list(self.X_data.columns).index(feature)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_vals,
            self.X_data,
            show=False
        )
        plt.tight_layout()
        plt.show()

    def plot_force_plot(self, instance_idx: int = 0) -> None:
        """Plot force plot for a single prediction."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        # Handle multi-output case
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values

        shap.force_plot(
            self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list)
            else self.explainer.expected_value,
            shap_vals[instance_idx],
            self.X_data.iloc[instance_idx],
            show=False
        )

    def get_prediction_explanations(self, instance_idx: int) -> pd.DataFrame:
        """Get feature contributions for a specific prediction."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        # Handle multi-output case
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values

        explanation_df = pd.DataFrame({
            'feature': self.X_data.columns,
            'feature_value': self.X_data.iloc[instance_idx].values,
            'shap_value': shap_vals[instance_idx]
        }).sort_values('shap_value', key=abs, ascending=False)

        return explanation_df
