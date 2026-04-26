# Trade Exception Predictor - Project Spec

## Project Overview

Building a machine learning pipeline to predict trade exceptions in financial trading using XGBoost with automated hyperparameter optimization, experiment tracking, and model explainability.

## Tech Stack

- **Framework**: XGBoost 2.0+
- **Hyperparameter Optimization**: Optuna
- **Experiment Tracking**: MLflow
- **Explainability**: SHAP
- **Cloud**: Azure ML
- **Data Processing**: Pandas, NumPy
- **ML**: scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Notebooks**: Jupyter

## Architecture

### Feature Engineering (`src/features.py`)
- Categorical encoding with LabelEncoder
- Interaction feature creation:
  - `volatility_risk` = market_volatility × counterparty_risk_score
  - `speed_deviation` = execution_speed_ms × price_deviation_pct
  - `size_volatility` = trade_size_percentile × market_volatility
- StandardScaler normalization
- Reusable transformer for inference

### Model Training (`src/train.py`)
- XGBoost binary classifier
- Optuna hyperparameter optimization:
  - Grid search over 9 hyperparameters
  - Early stopping with AUC metric
  - TPE sampler with median pruner
- MLflow integration:
  - Parameter logging
  - Metrics tracking
  - Model artifact storage
- Evaluation on test set with multiple metrics

### Explainability (`src/explain.py`)
- SHAP TreeExplainer for XGBoost
- Feature importance from mean absolute SHAP values
- Summary plots (bar, beeswarm)
- Dependence plots for feature interactions
- Force plots for individual predictions
- Instance-level explanation DataFrames

## Data Specification

### Input Format
- CSV with 13 features + target
- 10 synthetic trades for demo
- Imbalanced dataset: 60% normal, 40% exceptions

### Features
- **Categorical**: counterparty, instrument_type (encoded)
- **Numeric**: notional_amount, trade_price, market_volatility, counterparty_risk_score, execution_speed_ms, price_deviation_pct, trade_size_percentile
- **Datetime**: timestamp, settlement_date (dropped before modeling)
- **Target**: is_exception (binary: 0/1)

## Notebooks

### 01_eda.ipynb
- Data loading and inspection
- Exception distribution analysis
- Feature statistics and correlations
- Exception vs normal trades comparison
- Visualization of key patterns

### 02_modeling.ipynb
- Data preparation with FeatureEngineering
- Train-test split (80-20, stratified)
- Model training with Optuna (10 trials default)
- MLflow experiment tracking
- Evaluation metrics and plots
- SHAP explainability analysis

## Key Classes

### FeatureEngineering
```python
fe = FeatureEngineering(df)
X, y = fe.engineer_features()
X_new = fe.transform(X_new)  # Inference
```

### TradeExceptionPredictor
```python
predictor = TradeExceptionPredictor(n_trials=10)
predictor.train(X_train, y_train, X_test, y_test)
y_pred = predictor.predict(X_test)
y_proba = predictor.predict_proba(X_test)
```

### ExplainabilityAnalyzer
```python
analyzer = ExplainabilityAnalyzer(model, X_test, y_test)
importance_df = analyzer.get_feature_importance()
analyzer.plot_shap_summary()
analyzer.plot_shap_dependence('market_volatility')
```

## MLflow Tracking

- URI: `./mlflow_tracking/`
- Experiment: `trade_exception_predictor`
- Logged Parameters: All Optuna-optimized hyperparameters
- Logged Metrics: 
  - best_auc (from optimization)
  - test_auc, test_precision, test_recall, test_f1 (from evaluation)
- Logged Artifacts: Model artifact via `mlflow.xgboost.log_model()`

## Development Workflow

1. **Local Exploration**: Run EDA notebook
2. **Training**: Run modeling notebook
3. **Tracking**: View experiments with `mlflow ui`
4. **Deployment**: Save model from MLflow registry

## Expected Performance

- AUC target: >0.85 (exception detection)
- Precision: Balance false positives vs recalls
- Recall: Catch majority of exceptions
- Feature importance: Top 5 drive 60%+ of predictions

## Future Enhancements

- Azure ML pipeline integration
- Model endpoint deployment
- Batch scoring capability
- Production monitoring dashboards
- Additional synthetic data generation
- Ensemble methods
