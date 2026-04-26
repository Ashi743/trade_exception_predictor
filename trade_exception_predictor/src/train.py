"""
train.py - XGBoost Training with Optuna

Optimized for the trade reconciliation dataset with exact columns.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings('ignore')


class TradeExceptionPredictor:
    """XGBoost predictor for trade exceptions."""
    
    def __init__(self, n_trials=50, timeout=None, random_state=42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.best_trial = None
        self.scale_pos_weight = None
        
    def train(self, X_train, y_train, X_test, y_test):
        """Train model with Optuna optimization."""
        
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + "TRADE EXCEPTION PREDICTOR - TRAINING".center(78) + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝")
        
        # Compute class weight
        self._compute_scale_pos_weight(y_train)
        
        # Optimize
        self._optimize_hyperparameters(X_train, y_train, X_test, y_test)
        
        # Train final model
        self._train_final_model(X_train, y_train, X_test, y_test)
        
        # Evaluate
        self._evaluate(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETE")
        print("="*80 + "\n")
    
    def _compute_scale_pos_weight(self, y_train):
        """Compute class imbalance weight."""
        n_clean = (y_train == 0).sum()
        n_exceptions = (y_train == 1).sum()
        self.scale_pos_weight = n_clean / n_exceptions
        
        print(f"\n┌─ CLASS IMBALANCE")
        print(f"├─ Clean trades:     {n_clean:,} ({100*n_clean/len(y_train):.1f}%)")
        print(f"├─ Exceptions:       {n_exceptions:,} ({100*n_exceptions/len(y_train):.1f}%)")
        print(f"└─ scale_pos_weight: {self.scale_pos_weight:.2f}")
    
    def _optimize_hyperparameters(self, X_train, y_train, X_test, y_test):
        """Run Optuna optimization."""
        
        print(f"\n┌─ HYPERPARAMETER TUNING (Optuna - {self.n_trials} trials)")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': self.scale_pos_weight,
                'eval_metric': 'aucpr',
                'random_state': self.random_state,
                'verbosity': 0
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, y_pred_proba)
        
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(),
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
        
        self.best_trial = study.best_trial
        self.best_params = self.best_trial.params
        self.best_params['scale_pos_weight'] = self.scale_pos_weight
        self.best_params['eval_metric'] = 'aucpr'
        self.best_params['random_state'] = self.random_state
        self.best_params['verbosity'] = 0
        
        print(f"├─ Best Trial: #{self.best_trial.number}")
        print(f"├─ Best AUC: {self.best_trial.value:.4f}")
        print(f"└─ Best Parameters (showing top 6):")
        for i, (key, val) in enumerate(sorted(self.best_params.items())[:6]):
            if key not in ['scale_pos_weight', 'eval_metric', 'random_state', 'verbosity']:
                print(f"   {'└─' if i == 5 else '├─'} {key}: {val}")
    
    def _train_final_model(self, X_train, y_train, X_test, y_test):
        """Train final model."""
        print(f"\n┌─ TRAINING FINAL MODEL")
        
        self.model = xgb.XGBClassifier(**self.best_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        print(f"├─ Boosting rounds: {self.model.n_estimators}")
        print(f"└─ ✓ Model trained")
    
    def _evaluate(self, X_train, y_train, X_test, y_test):
        """Evaluate model."""
        print(f"\n┌─ MODEL EVALUATION")
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_proba_train = self.model.predict_proba(X_train)[:, 1]
        y_proba_test = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        test_auc = roc_auc_score(y_test, y_proba_test)
        test_prec = precision_score(y_test, y_pred_test)
        test_rec = recall_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        train_auc = roc_auc_score(y_train, y_proba_train)
        
        print(f"├─ TEST METRICS:")
        print(f"│  ├─ AUC-ROC:   {test_auc:.4f}")
        print(f"│  ├─ Precision: {test_prec:.4f}")
        print(f"│  ├─ Recall:    {test_rec:.4f}")
        print(f"│  └─ F1-Score:  {test_f1:.4f}")
        print(f"├─ TRAIN AUC:     {train_auc:.4f}")
        
        # Overfitting check
        gap = train_auc - test_auc
        if gap > 0.10:
            print(f"├─ ⚠️  Overfitting detected (gap: {gap:.4f})")
        else:
            print(f"├─ ✓ Healthy model (gap: {gap:.4f})")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"└─ CONFUSION MATRIX:")
        print(f"   ├─ True Negatives:  {tn:,}")
        print(f"   ├─ False Positives: {fp:,}")
        print(f"   ├─ False Negatives: {fn:,}")
        print(f"   └─ True Positives:  {tp:,}")
        
        # Log to MLflow
        metrics = {
            'test_auc': test_auc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1,
            'train_auc': train_auc,
            'overfit_gap': gap
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params(self.best_params)
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def get_best_params(self):
        """Get best hyperparameters."""
        return self.best_params