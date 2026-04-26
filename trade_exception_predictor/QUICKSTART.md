# Trade Exception Predictor - Quick Start Guide

## Project Files at a Glance

```
trade_exception_predictor/
├── src/                    ← Core ML modules
│   ├── generate_data.py    ← Create synthetic training data
│   ├── features.py         ← Feature engineering (encode, scale)
│   ├── train.py            ← XGBoost + Optuna training
│   ├── explain.py          ← SHAP explainability
│   └── score.py            ← Azure inference service
├── azure/                  ← Cloud deployment scripts
│   ├── submit_job.py       ← Submit to Azure ML
│   ├── register_model.py   ← Register in model registry
│   └── deploy_endpoint.py  ← Deploy REST endpoint
├── notebooks/              ← Interactive exploration
│   ├── 01_eda.ipynb        ← Data exploration
│   └── 02_modeling.ipynb   ← Full training pipeline
├── data/
│   └── trades_synthetic.csv ← Training data (auto-generated)
├── mlflow_tracking/        ← Experiment logs (auto-created)
├── ARCHITECTURE.md         ← Detailed dependency mapping
├── DEPENDENCIES.txt        ← Visual flow diagrams
└── QUICKSTART.md           ← This file
```

---

## 1️⃣ LOCAL DEVELOPMENT (5 minutes)

### Step 1: Generate Data
```bash
cd trade_exception_predictor
python src/generate_data.py
# Creates: data/trades_synthetic.csv (1000 trades)
```

### Step 2: Explore Data
```bash
jupyter notebook notebooks/01_eda.ipynb
# Browse through cells
# See: exception distribution, feature stats, correlations
```

### Step 3: Train Model
```bash
jupyter notebook notebooks/02_modeling.ipynb
# Run all cells in order:
#   1. Imports
#   2. Load data
#   3. Feature engineering
#   4. Train-test split
#   5. Model training (5-10 min)
#   6. Evaluation metrics
#   7. SHAP analysis
```

### Step 4: View Results
```bash
mlflow ui --backend-store-uri ./mlflow_tracking
# Open: http://localhost:5000
# Compare experiments, view metrics
```

---

## 2️⃣ KEY FILE RELATIONSHIPS

### Data Flow: Raw → Predictions
```
trades_synthetic.csv
    ↓ [read]
features.py::FeatureEngineering
    ↓ [transform]
X (scaled features) + y (labels)
    ↓ [train]
train.py::TradeExceptionPredictor
    ↓ [evaluate]
explain.py::ExplainabilityAnalyzer
    ↓ [save]
mlflow_tracking/ (model + metadata)
```

### Critical Imports (Must Have These!)
```python
# Notebooks use these:
from features import FeatureEngineering
from train import TradeExceptionPredictor
from explain import ExplainabilityAnalyzer

# Score.py uses this:
from features import FeatureEngineering  # ← MUST HAVE for inference
```

---

## 3️⃣ FILE PURPOSES CHEAT SHEET

| File | What It Does | When to Use |
|------|-------------|------------|
| **src/generate_data.py** | Creates 1000+ synthetic trades | First time setup |
| **src/features.py** | Encodes, scales, creates interactions | Always (core dependency) |
| **src/train.py** | XGBoost + Optuna hyperparameter tuning | Training |
| **src/explain.py** | SHAP feature importance & plots | After training |
| **src/score.py** | Inference for Azure endpoint | Deployment |
| **notebooks/01_eda.ipynb** | Data exploration & visualization | Data understanding |
| **notebooks/02_modeling.ipynb** | Full pipeline: train → evaluate → explain | Main workflow |
| **azure/submit_job.py** | Send training to Azure ML cluster | Cloud training |
| **azure/register_model.py** | Save model to Azure registry | After cloud training |
| **azure/deploy_endpoint.py** | Create REST API endpoint | Production deployment |

---

## 4️⃣ QUICK EXECUTION GUIDE

### Local Only (Fastest)
```bash
# 1. Generate data
python src/generate_data.py

# 2-3. Run notebooks (interactive)
jupyter notebook notebooks/02_modeling.ipynb

# 4. View results
mlflow ui --backend-store-uri ./mlflow_tracking
```

### Azure Cloud (Full Pipeline)
```bash
# Get Azure info from Azure Portal:
# - Subscription ID
# - Resource Group name
# - ML Workspace name

# 1. Submit training to cloud
python azure/submit_job.py <sub_id> <resource_group> <workspace>
# Takes 5-10 minutes... wait for completion

# 2. Get the MLflow run ID from Azure output
# Look for: runs:/<run_id>

# 3. Register the trained model
python azure/register_model.py <sub> <rg> <ws> <run_id> trade-exception-model 1.0

# 4. Deploy as REST endpoint
python azure/deploy_endpoint.py <sub> <rg> <ws> trade-exception-model 1.0

# 5. Test the endpoint (returns REST URI)
curl -X POST <scoring_uri>/score \
  -H "Content-Type: application/json" \
  -d '{"counterparty": "Bank_A", "notional_amount": 1000000, ...}'
```

---

## 5️⃣ UNDERSTANDING THE PIPELINE

### What Each Layer Does:

**Layer 1: Features** (src/features.py)
- Input: Raw DataFrame from CSV
- Process: Encode categories, create 3 interaction features, scale
- Output: X (clean features), y (labels)
- Used by: Everything downstream

**Layer 2: Training** (src/train.py)
- Input: X_train, y_train, X_test, y_test
- Process: Optuna optimization, XGBoost training, evaluation
- Output: Trained model + MLflow logs
- Quality: ROC-AUC ~0.85+ expected

**Layer 3: Explanation** (src/explain.py)
- Input: Trained model + test data
- Process: SHAP values, feature importance
- Output: Visualizations, explanation DataFrames
- Use: Understand what model learned

**Layer 4: Inference** (src/score.py)
- Input: score.py + features.py + model.pkl
- Process: Load model, transform new data, predict
- Output: JSON {prediction, probabilities}
- Deploy: On Azure ML endpoint

---

## 6️⃣ DEPENDENCY TREE (What Needs What)

```
✓ generate_data.py
  └─ (no dependencies, standalone)

✓ features.py
  ├─ pandas, numpy
  └─ sklearn.preprocessing

✓ train.py
  ├─ xgboost
  ├─ optuna
  ├─ mlflow
  └─ sklearn.metrics

✓ explain.py
  ├─ shap
  ├─ pandas, numpy
  └─ matplotlib

✓ score.py
  ├─ features.py ★ CRITICAL
  ├─ joblib
  └─ json, pandas

✓ notebooks/02_modeling.ipynb
  ├─ features.py ★
  ├─ train.py ★
  ├─ explain.py ★
  └─ sklearn, matplotlib

✓ azure/submit_job.py
  ├─ train.py (remote)
  └─ azure.ai.ml, azure.identity

✓ azure/register_model.py
  ├─ mlflow
  └─ azure.ai.ml

✓ azure/deploy_endpoint.py
  ├─ score.py ★
  ├─ features.py ★ (via score.py)
  ├─ environment.yml
  └─ azure.ai.ml

★ = Critical (cannot work without)
```

---

## 7️⃣ TROUBLESHOOTING

### "ImportError: No module named 'src'"
**Solution**: Run notebooks from `notebooks/` directory or use `sys.path.insert(0, '../src')`

### "features.py not found in score.py"
**Solution**: Ensure `features.py` is in same directory as `score.py` when deployed

### "MLflow URI not found"
**Solution**: Use absolute path: `mlflow.set_tracking_uri('/full/path/to/mlflow_tracking')`

### "Azure authentication failed"
**Solution**: Run `az login` first, ensure proper Azure credentials

### "Model accuracy too low"
**Solution**: Check feature engineering in `features.py` - may need more interaction features

---

## 8️⃣ WHAT EACH FILE OUTPUTS

| File | Outputs |
|------|---------|
| generate_data.py | `data/trades_synthetic.csv` (CSV) |
| 01_eda.ipynb | Plots, statistics (display only) |
| 02_modeling.ipynb | Plots, metrics (display) + MLflow logs |
| train.py | `mlflow_tracking/` (experiments, models) |
| explain.py | SHAP plots (display only) |
| azure/submit_job.py | Azure job ID, status messages |
| azure/register_model.py | Model registry entry |
| azure/deploy_endpoint.py | REST API endpoint URI |

---

## 9️⃣ TYPICAL RUNTIMES

| Task | Duration |
|------|----------|
| generate_data.py | <1 second |
| 01_eda.ipynb | 2-3 minutes (display plots) |
| 02_modeling.ipynb | 5-10 minutes (Optuna optimization) |
| azure/submit_job.py | 5-10 minutes (cloud training) |
| Full pipeline (local) | 10-15 minutes |
| Full pipeline (Azure) | 20-30 minutes |

---

## 🔟 NEXT STEPS CHECKLIST

- [ ] Read ARCHITECTURE.md for deep dive
- [ ] Read DEPENDENCIES.txt for visual diagrams
- [ ] Run `python src/generate_data.py`
- [ ] Open `notebooks/01_eda.ipynb` in Jupyter
- [ ] Open `notebooks/02_modeling.ipynb` in Jupyter
- [ ] Run all cells in 02_modeling.ipynb
- [ ] Check MLflow UI at localhost:5000
- [ ] Review SHAP plots in notebook
- [ ] (Optional) Deploy to Azure using azure/* scripts

---

## 📚 Documentation Files in Order

1. **QUICKSTART.md** ← You are here
2. **ARCHITECTURE.md** ← Deep dive into design
3. **DEPENDENCIES.txt** ← Visual flow diagrams
4. **README.md** ← Project overview
5. **CLAUDE.md** ← Technical specifications

---

## 🚀 One-Liner Commands

```bash
# Full local pipeline
python src/generate_data.py && jupyter notebook notebooks/02_modeling.ipynb && mlflow ui --backend-store-uri ./mlflow_tracking

# Full Azure pipeline
python azure/submit_job.py $SUB $RG $WS && python azure/register_model.py $SUB $RG $WS <run_id> && python azure/deploy_endpoint.py $SUB $RG $WS trade-exception-model 1.0
```

---

Happy coding! 🎉
