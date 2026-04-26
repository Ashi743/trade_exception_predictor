"""XGBoost model training with Optuna hyperparameter optimization."""
try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("xgboost is not installed. Please run 'pip install xgboost' to use this module.") from e
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import mlflow
from typing import Optional, Tuple


class TradeExceptionPredictor:
    """XGBoost predictor with Optuna optimization."""

    def __init__(self, n_trials: int = 50, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_names = None

    def objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:
        """Optuna objective function."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': self.random_state,
            'verbosity': 0,
            'eval_metric': 'logloss'
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)

        return auc_score

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> None:
        """Train model with hyperparameter optimization."""
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )

        self.feature_names = X_train.columns.tolist()

        # Optuna optimization
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner()
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

        study.optimize(
            lambda trial: self.objective(trial, X_tr, y_tr, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        mlflow.log_params(self.best_params)
        mlflow.log_metric('best_auc', study.best_value)

        # Train final model with best params
        self.best_params['random_state'] = self.random_state
        self.best_params['verbosity'] = 0
        self.model = xgb.XGBClassifier(**self.best_params)
        self.model.fit(X_train, y_train)

        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            self._evaluate(X_test, y_test)

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluate model on test set."""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_pred, zero_division=0)
        }

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return feature_importance_df
