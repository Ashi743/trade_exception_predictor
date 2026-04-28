# Trade Exception Predictor

A machine learning solution for predicting trade exceptions using XGBoost with Optuna hyperparameter optimization, MLflow experiment tracking, and SHAP explainability analysis.

## Features

- **XGBoost Classification**: Fast and scalable gradient boosting for binary classification
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **MLflow Integration**: Experiment tracking and model management
- **SHAP Explainability**: Understand model predictions with SHAP values
- **Azure ML Integration**: Deploy to Azure ML endpoints

## Project Structure

```
trade_exception_predictor/
├── data/
│   └── trades_synthetic.csv         # Synthetic trading data
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   └── 02_modeling.ipynb            # Model training and evaluation
├── src/
│   ├── features.py                  # Feature engineering
│   ├── train.py                     # XGBoost + Optuna training
│   └── explain.py                   # SHAP explainability
├── mlflow_tracking/                 # MLflow experiment logs
├── requirements.txt
└── README.md
```

## Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development

1. **Exploratory Data Analysis**:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

2. **Model Training**:
```bash
jupyter notebook notebooks/02_modeling.ipynb
```

3. **View MLflow Dashboard**:
```bash
mlflow ui --backend-store-uri ./mlflow_tracking
```

## Data Features

- `trade_id`: Unique trade identifier
- `timestamp`: Trade execution time
- `counterparty`: Trading counterparty
- `instrument_type`: FX, EQUITY, BOND
- `notional_amount`: Trade value
- `trade_price`: Execution price
- `settlement_date`: Settlement date
- `market_volatility`: Market volatility indicator
- `counterparty_risk_score`: Counterparty credit risk
- `execution_speed_ms`: Execution latency
- `price_deviation_pct`: Price deviation from market
- `trade_size_percentile`: Trade size percentile
- `is_exception`: Target label (0/1)

## Model Architecture

The model uses:
- XGBoost classifier with optimized hyperparameters
- Engineered features including interaction terms
- Standardized numeric features
- Encoded categorical features

## Performance Metrics

Tracked in MLflow:
- ROC-AUC Score
- Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importances

## Explainability

SHAP values provide:
- Feature importance rankings
- Individual prediction explanations
- Feature dependence plots
- Summary force plots

## Requirements

Key dependencies:
- xgboost >= 2.0.0
- optuna >= 3.0.0
- mlflow >= 2.0.0
- shap >= 0.42.0
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- jupyter

See `requirements.txt` for complete list.

## MLflow Tracking

All experiments are logged to `mlflow_tracking/`. To view results:

```bash
mlflow ui --backend-store-uri ./mlflow_tracking
```

Access at http://localhost:5000

## License

MIT
    