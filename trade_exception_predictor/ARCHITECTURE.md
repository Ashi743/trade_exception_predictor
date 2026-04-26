# Trade Exception Predictor - Architecture & Dependencies

## Project Structure
```
trade_exception_predictor/
├── data/
│   └── trades_synthetic.csv              # Raw input data (15k trades)
├── src/
│   ├── generate_data.py                  # [STANDALONE] Generate synthetic data
│   ├── features.py                       # [CORE] Feature engineering pipeline
│   ├── train.py                          # [CORE] Model training with Optuna
│   ├── explain.py                        # [CORE] SHAP explainability analysis
│   └── score.py                          # [DEPLOYMENT] Inference service
├── azure/
│   ├── submit_job.py                     # [AZURE] Submit training to cloud
│   ├── register_model.py                 # [AZURE] Register to model registry
│   └── deploy_endpoint.py                # [AZURE] Deploy managed endpoint
├── notebooks/
│   ├── 01_eda.ipynb                      # [INTERACTIVE] Data exploration
│   └── 02_modeling.ipynb                 # [INTERACTIVE] Training pipeline
└── mlflow_tracking/                      # Experiment logs (auto-created)
```

---

## File Dependencies & Data Flow

### 1. DATA GENERATION & PREPARATION
```
generate_data.py (STANDALONE)
  ├─ Input: None (synthetic generation)
  ├─ Output: data/trades_synthetic.csv
  └─ Dependencies: pandas, numpy, datetime
```

### 2. FEATURE ENGINEERING (Core Pipeline)
```
features.py
  ├─ Class: FeatureEngineering
  ├─ Input: pd.DataFrame (raw trades)
  ├─ Methods:
  │   ├─ engineer_features() → (X: DataFrame, y: Series)
  │   └─ transform(X) → transformed DataFrame
  ├─ Operations:
  │   ├─ Drop: trade_id, timestamp, settlement_date
  │   ├─ Encode: categorical columns (LabelEncoder)
  │   ├─ Create interactions:
  │   │   ├─ volatility_risk = market_volatility × counterparty_risk_score
  │   │   ├─ speed_deviation = execution_speed_ms × price_deviation_pct
  │   │   └─ size_volatility = trade_size_percentile × market_volatility
  │   └─ Scale: StandardScaler normalization
  └─ Dependencies: pandas, numpy, sklearn.preprocessing
```

### 3. MODEL TRAINING (Core Pipeline)
```
train.py
  ├─ Class: TradeExceptionPredictor
  ├─ Input: X_train, y_train, X_test, y_test
  ├─ Dependencies: features.py (indirect - used in notebooks)
  ├─ Hyperparameter Optimization:
  │   ├─ Method: Optuna with TPE Sampler
  │   ├─ Metric: ROC-AUC (validation set)
  │   ├─ Grid: max_depth, learning_rate, n_estimators, subsample, etc.
  │   └─ Early stopping: 10 rounds patience
  ├─ Training:
  │   ├─ Validation split: 80-20 from training
  │   ├─ Model: xgboost.XGBClassifier
  │   └─ Evaluation: AUC, Precision, Recall, F1
  ├─ MLflow Logging:
  │   ├─ Hyperparameters (best_params)
  │   ├─ Metrics (best_auc, test_auc, test_precision, test_recall, test_f1)
  │   └─ Model artifact (model.pkl)
  └─ Dependencies: xgboost, optuna, mlflow, sklearn.metrics, sklearn.model_selection
```

### 4. EXPLAINABILITY (Analysis Pipeline)
```
explain.py
  ├─ Class: ExplainabilityAnalyzer
  ├─ Input: model (XGBClassifier), X_data, y_data (optional)
  ├─ Methods:
  │   ├─ get_feature_importance() → DataFrame
  │   ├─ plot_shap_summary(plot_type='bar')
  │   ├─ plot_shap_dependence(feature_name)
  │   ├─ plot_force_plot(instance_idx)
  │   └─ get_prediction_explanations(instance_idx) → DataFrame
  ├─ SHAP Analysis:
  │   ├─ TreeExplainer for XGBoost
  │   ├─ Mean absolute SHAP values = feature importance
  │   └─ Individual prediction contributions
  └─ Dependencies: shap, pandas, numpy, matplotlib
```

### 5. INFERENCE SERVICE (Deployment Pipeline)
```
score.py
  ├─ Class: TradeExceptionScoringService
  ├─ Input: model_path (trained XGBoost), encoder_path (FeatureEngineering)
  ├─ Dependencies: features.py (REQUIRED - for transform())
  ├─ Methods:
  │   ├─ run(raw_data) → JSON predictions
  │   └─ init() for Azure ML initialization
  ├─ Azure Endpoint Integration:
  │   ├─ Loads saved model and feature_engineer
  │   ├─ Transforms new data same way as training
  │   ├─ Returns: {prediction, probability_normal, probability_exception}
  │   └─ Error handling with JSON responses
  └─ Dependencies: pandas, numpy, json, joblib, features.py
```

### 6. AZURE CLOUD INTEGRATION

#### 6a. Submit Training Job
```
azure/submit_job.py
  ├─ Function: submit_training_job()
  ├─ Input: subscription_id, resource_group, workspace_name
  ├─ Triggers: Remote training on Azure ML compute
  ├─ Reads: src/train.py (runs on cluster)
  ├─ Reads: data/trades_synthetic.csv (input data)
  ├─ Outputs: Model artifact in Azure ML
  └─ Dependencies: azure.ai.ml, azure.identity
```

#### 6b. Register Model
```
azure/register_model.py
  ├─ Function: register_model_from_mlflow()
  ├─ Input: mlflow_run_id, model_name, version
  ├─ Links: MLflow run → Azure ML Model Registry
  ├─ Metadata: AUC, F1, framework (xgboost), tags
  └─ Dependencies: azure.ai.ml, mlflow
```

#### 6c. Deploy Endpoint
```
azure/deploy_endpoint.py
  ├─ Function: deploy_model_endpoint()
  ├─ Input: model_name, model_version, endpoint_name
  ├─ Creates: ManagedOnlineEndpoint
  ├─ Deployment:
  │   ├─ Runs: src/score.py (inference script)
  │   ├─ Uses: environment.yml (conda env)
  │   ├─ Compute: Standard_DS2_v2 (1 instance)
  │   └─ Scoring URI: REST API endpoint
  ├─ Testing: Invoke endpoint with sample data
  └─ Dependencies: azure.ai.ml, score.py, environment.yml
```

---

## Notebooks - Execution Flow

### 01_eda.ipynb (Exploratory)
```
1. Load: data/trades_synthetic.csv
2. Display: shape, dtypes, missing values
3. Analyze: exception distribution, statistics
4. Visualize: correlation matrix, exception patterns
5. Output: Static plots, summary statistics
```

### 02_modeling.ipynb (Pipeline)
```
1. Import: sys.path.insert(0, '../src')
   ├─ from features import FeatureEngineering
   ├─ from train import TradeExceptionPredictor
   └─ from explain import ExplainabilityAnalyzer
2. Load: data/trades_synthetic.csv
3. Feature Engineering:
   └─ fe = FeatureEngineering(df)
   └─ X, y = fe.engineer_features()
4. Split: train_test_split(X, y, test_size=0.2)
5. Train:
   └─ predictor = TradeExceptionPredictor(n_trials=10)
   └─ predictor.train(X_train, y_train, X_test, y_test)
6. Evaluate:
   ├─ y_pred = predictor.predict(X_test)
   ├─ y_proba = predictor.predict_proba(X_test)
   ├─ Metrics: classification_report, roc_auc, confusion_matrix
   └─ Plots: ROC curve, confusion matrix heatmap
7. Explain:
   └─ analyzer = ExplainabilityAnalyzer(model, X_test, y_test)
   ├─ analyzer.get_feature_importance()
   ├─ analyzer.plot_shap_summary()
   └─ analyzer.plot_shap_dependence('market_volatility')
8. MLflow Logging:
   └─ All metrics and model automatically logged
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

Raw Data (CSV)
    │
    ├─→ features.py::FeatureEngineering
    │       ├─ Encode categories
    │       ├─ Create interactions
    │       └─ StandardScale
    │
    ├─→ train.py::TradeExceptionPredictor
    │       ├─ Optuna optimization (10-50 trials)
    │       ├─ Train XGBoost
    │       ├─ Evaluate metrics
    │       └─ Log to MLflow
    │
    └─→ explain.py::ExplainabilityAnalyzer
            ├─ SHAP TreeExplainer
            ├─ Feature importance
            └─ Generate plots

┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PIPELINE                           │
└─────────────────────────────────────────────────────────────────┘

MLflow Model Registry
    │
    ├─→ azure/register_model.py
    │       └─ Register to Azure ML
    │
    └─→ azure/deploy_endpoint.py
            ├─ Create endpoint
            ├─ Load src/score.py
            ├─ Deploy FeatureEngineering
            └─ REST API ready

┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

New Trade Data (JSON/CSV)
    │
    └─→ score.py::TradeExceptionScoringService
            ├─ Load: model.pkl + feature_engineer.pkl
            ├─ Transform: features.py::FeatureEngineering
            ├─ Predict: xgboost.predict_proba()
            └─ Return: JSON {prediction, probabilities}
```

---

## Key Relationships

### Direct Dependencies (Import)
| File | Imports | Used By |
|------|---------|---------|
| `features.py` | pandas, numpy, sklearn | `train.py`, `score.py`, notebooks |
| `train.py` | features (indirect), xgboost, optuna, mlflow | notebooks, azure jobs |
| `explain.py` | shap, pandas, numpy, matplotlib | `02_modeling.ipynb` |
| `score.py` | features.py (DIRECT), joblib | `deploy_endpoint.py` |
| `generate_data.py` | pandas, numpy, datetime | Manual execution |
| `azure/submit_job.py` | azure.ai.ml, train.py (remote) | Manual: VSCode |
| `azure/register_model.py` | azure.ai.ml, mlflow | Manual: Post-training |
| `azure/deploy_endpoint.py` | azure.ai.ml, score.py | Manual: Final deployment |

### Data Flow Dependencies
```
generate_data.py → trades_synthetic.csv
    ↓
features.py ← ← ← ← notebooks (01_eda.ipynb, 02_modeling.ipynb)
    ↓
train.py ← ← ← ← 02_modeling.ipynb
    ↓
MLflow logs
    ↓
register_model.py ← ← ← azure command
    ↓
deploy_endpoint.py ← ← ← azure command
    ↓
score.py + features.py → Azure endpoint
    ↓
REST API predictions
```

---

## Execution Order (Recommended)

### Local Development (Step-by-Step)
1. `python src/generate_data.py` — Create training data
2. `jupyter notebook notebooks/01_eda.ipynb` — Explore data
3. `jupyter notebook notebooks/02_modeling.ipynb` — Train model
4. Check `mlflow ui --backend-store-uri ./mlflow_tracking` — View experiments

### Azure Deployment
1. `python azure/submit_job.py <sub_id> <rg> <ws>` — Train on Azure
2. `python azure/register_model.py <sub_id> <rg> <ws> <run_id>` — Register model
3. `python azure/deploy_endpoint.py <sub_id> <rg> <ws> <model_name> <version>` — Deploy endpoint
4. Test endpoint with sample predictions

---

## Configuration Files

| File | Purpose | Key Settings |
|------|---------|--------------|
| `environment.yml` | Conda environment for Azure cluster | Python 3.11, pip packages |
| `requirements.txt` | pip dependencies for local dev | xgboost, optuna, mlflow, azure-ai-ml |
| `.gitignore` | Exclude from git | `data/*.csv`, `mlruns/`, `models/` |

---

## Key Classes & Methods

### FeatureEngineering
- `__init__(df)` — Initialize with DataFrame
- `engineer_features()` → (X, y) — Fit and transform
- `transform(X)` → X_transformed — Apply fitted transformations

### TradeExceptionPredictor
- `__init__(n_trials=50)` — Initialize with Optuna config
- `train(X_train, y_train, X_test, y_test)` — Optimize and train
- `predict(X)` → predictions
- `predict_proba(X)` → probabilities
- `get_feature_importance()` → DataFrame

### ExplainabilityAnalyzer
- `__init__(model, X_data, y_data)` — Initialize with trained model
- `get_feature_importance()` → SHAP-based importance
- `plot_shap_summary()` — Summary plot
- `plot_shap_dependence(feature)` — Feature dependence
- `get_prediction_explanations(idx)` → Instance explanation

### TradeExceptionScoringService
- `__init__(model_path, encoder_path)` — Load artifacts
- `run(raw_data)` → JSON predictions for Azure endpoint
- `init()` — Azure ML initialization hook
