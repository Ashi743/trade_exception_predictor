# Trade Exception Predictor - Project Knowledge Map

**Generated:** 2026-04-28  
**Project Type:** Machine Learning Pipeline for Financial Risk Detection  
**Status:** Active Development

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Input (Trade Records)                   │
│                   25 Features + 1 Target (Binary)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│           FeatureEngineering (src/features.py)                  │
│  ├─ Date Parsing (trade_date, settlement_date)                 │
│  ├─ Categorical Encoding                                        │
│  │  ├─ One-Hot: commodity_type, instrument_type, etc.          │
│  │  └─ Label Encoding: counterparty_id, counterparty_tier       │
│  ├─ Interaction Features (5 engineered features)                │
│  ├─ StandardScaler Normalization (numeric features)             │
│  └─ Train-Test Split (80-20, stratified)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│        TradeExceptionPredictor (src/train.py)                    │
│  ├─ Class Weight Computation (scale_pos_weight)                 │
│  ├─ Hyperparameter Tuning (Optuna, 50 trials)                   │
│  │  ├─ max_depth: [3, 10]                                       │
│  │  ├─ learning_rate: [0.01, 0.3]                               │
│  │  ├─ n_estimators: [100, 500]                                 │
│  │  └─ 4 more hyperparameters (subsample, colsample, etc.)      │
│  ├─ Model Training (XGBoost binary classifier)                  │
│  └─ Evaluation (AUC, Precision, Recall, F1, confusion matrix)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│       ExplainabilityAnalyzer (src/explain.py)                   │
│  ├─ SHAP TreeExplainer Initialization                           │
│  ├─ Feature Importance Ranking (mean |SHAP values|)             │
│  ├─ Per-Instance Explanations (waterfall, force plots)          │
│  ├─ Dependence Plots (feature interactions)                     │
│  └─ High-Risk Trade Analysis                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│    TradeExceptionScoringService (src/score.py)                  │
│  ├─ Model Loading (pkl files)                                   │
│  ├─ Single Trade Scoring (real-time inference)                  │
│  ├─ Per-Trade SHAP Explanations                                 │
│  ├─ Risk Confidence Classification (high/medium/low)            │
│  ├─ Operational Recommendations (emoji-flagged)                 │
│  ├─ Batch Scoring Interface                                     │
│  ├─ Azure ML Endpoint Interface (init/run)                      │
│  └─ Local Testing Utilities                                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow

### Input Features (25 columns)

**Identifiers:**
- `trade_id`, `trade_date`, `settlement_date`

**Trade Characteristics:**
- `days_to_settlement`, `commodity_type`, `instrument_type`, `delivery_location`

**Counterparty Risk:**
- `counterparty_id`, `counterparty_tier`, `counterparty_region`
- `counterparty_exception_rate_30d` ← Historical exception propensity

**Trade Financials:**
- `notional_usd`, `quantity_mt`, `price_per_mt`, `settlement_currency`

**Market Conditions:**
- `market_volatility`, `price_volatility_flag`, `price_deviation_pct`

**Temporal Signals:**
- `is_month_end`, `is_quarter_end`, `day_of_week`

**Operational Flags:**
- `cross_border_flag`, `currency_mismatch_flag`
- `same_commodity_breaks_7d` ← Industry signal
- `documentation_lag_days`, `amendment_count` ← Operational friction

**Target:**
- `is_exception` (binary: 0=clean, 1=exception)

### Engineered Features (5 interaction features added)

| Feature | Logic | Why | Type |
|---------|-------|-----|------|
| `tier_x_month_end` | `counterparty_tier × is_month_end` | Small traders struggle under month-end stress | Multiplicative |
| `tier_x_doc_lag` | `counterparty_tier × documentation_lag_days` | Late docs from risky parties compound risk | Multiplicative |
| `high_doc_lag` | `documentation_lag_days > 3` | Binary flag for critical doc delays | Threshold |
| `high_amendments` | `amendment_count > 2` | Binary flag for excessive trade modifications | Threshold |
| `doc_lag_x_amendments` | `documentation_lag_days × amendment_count` | Combined operational friction | Multiplicative |

**Preprocessing Pipeline:**
1. Date parsing → datetime objects
2. Categorical encoding → 24 one-hot features + 3 label-encoded features
3. Interaction feature creation → 5 new features
4. StandardScaler normalization → all numeric features
5. Cleanup → remove identifiers, keep only numeric
6. **Result:** ~40 numeric features for modeling

---

## 🔗 Component Dependencies

### Cross-Module Relationships

```
features.py
  │
  ├─ Input: Raw CSV (25 columns)
  ├─ Output: X_train, X_test, y_train, y_test, feature_names
  └─ Classes: FeatureEngineering
      └─ Uses: sklearn (LabelEncoder, StandardScaler, train_test_split)

train.py
  │
  ├─ Input: X_train, X_test, y_train, y_test from features.py
  ├─ Output: Trained XGBoost model + best_params
  └─ Classes: TradeExceptionPredictor
      ├─ Uses: xgboost, optuna (TPESampler, MedianPruner), sklearn (metrics)
      └─ Integrates: MLflow (logs params, metrics, model artifact)

explain.py
  │
  ├─ Input: Trained model + X_test + y_test
  ├─ Output: Feature importance, per-instance explanations, plots
  └─ Classes: ExplainabilityAnalyzer
      ├─ Uses: shap (TreeExplainer), matplotlib
      └─ Depends on: model.predict_proba() interface

score.py
  │
  ├─ Input: Saved model.pkl + feature_engineer.pkl
  ├─ Output: JSON predictions + SHAP explanations
  └─ Classes: TradeExceptionScoringService
      ├─ Reuses: FeatureEngineering.transform() (inference-time feature application)
      ├─ Uses: shap (TreeExplainer), joblib
      └─ Provides: Azure ML endpoint interface (init/run)

notebooks/01_eda.ipynb
  │
  ├─ Input: Raw trade data CSV
  ├─ Output: Data understanding, feature correlations, class balance insights
  └─ Imports: FeatureEngineering from src/features.py (optional)

notebooks/02_modeling.ipynb
  │
  ├─ Input: Raw trade data CSV
  ├─ Imports: FeatureEngineering, TradeExceptionPredictor, ExplainabilityAnalyzer
  ├─ Workflow: EDA → Features → Train → Explain → Export to MLflow
  └─ Output: Trained model artifacts + training reports
```

---

## 🎯 Key Insights & Hidden Dependencies

### 1. **Feature Engineering is Stateful**
- `FeatureEngineering.engineer_features()` **fits** label encoders and scalers
- `FeatureEngineering.transform()` uses **fitted state** from `engineer_features()`
- **Implication:** Never call `transform()` before `engineer_features()` — causes `ValueError`
- **Production Impact:** The feature_engineer.pkl serializes this state; inference relies on exact same transformations

### 2. **Hyperparameter Tuning Depends on Data Imbalance**
- `scale_pos_weight` computed from y_train: `n_clean / n_exceptions`
- This weighting is **baked into every Optuna trial**
- Different datasets → different optimal hyperparameters
- **Surprise Connection:** Optuna's best parameters are only valid for the dataset class distribution they were tuned on

### 3. **SHAP Explanations Bridge Two Worlds**
- ExplainabilityAnalyzer expects `model.predict_proba()` interface (sklearn-style)
- TradeExceptionScoringService recomputes SHAP on same model
- **Opportunity:** SHAP values for test set could be cached to avoid recomputation in production

### 4. **Categorical Encoding Creates Feature Count Inflation**
- Input: 25 features
- After one-hot: ~24 new columns (5 categorical features × 4-5 categories each)
- After label encoding: +3 columns
- After interactions: +5 columns
- **Result:** ~40 final features → model training time scales with this explosion

### 5. **Operational Recommendations Embed Business Rules**
- `_generate_recommendation()` in score.py hardcodes confidence thresholds (0.3, 0.7, 0.9)
- These are **not tuned from the model** — they're manually chosen
- **Gap:** No A/B testing or calibration against actual exception resolution costs

---

## 🔄 Data Transformation Pipeline

```
RAW TRADE DATA
    ↓
[01_eda.ipynb] → Understand distributions, exceptions, correlations
    ↓
[FeatureEngineering.engineer_features()]
    ├─ Parse dates
    ├─ One-hot categorical (5 features → ~24 new columns)
    ├─ Label encode counterparty attributes (3 features)
    ├─ Create 5 interaction features
    ├─ StandardScaler normalize
    └─ Return: X_train, X_test, y_train, y_test, feature_names
    ↓
[02_modeling.ipynb → TradeExceptionPredictor.train()]
    ├─ Optuna hyperparameter tuning (50 trials)
    ├─ Train XGBoost with best params
    └─ MLflow tracking (params, metrics, model artifact)
    ↓
[ExplainabilityAnalyzer]
    ├─ SHAP TreeExplainer initialization
    ├─ Feature importance ranking
    └─ Per-instance explanations (waterfall, force, summary plots)
    ↓
[INFERENCE MODE] → score.py
    ├─ TradeExceptionScoringService loads model + feature_engineer
    ├─ Incoming trade → FeatureEngineering.transform()
    ├─ Model.predict_proba()
    ├─ SHAP explanation (per-trade)
    └─ Return: JSON with prediction + drivers + recommendation
```

---

## 🌉 Cross-Component Bridges

### Bridge 1: Feature Names
- **Source:** `FeatureEngineering.get_feature_names()` after fitting
- **Consumer:** `ExplainabilityAnalyzer` (uses for SHAP plot labels)
- **Consumer:** `TradeExceptionScoringService._get_top_drivers()` (maps SHAP values to feature names)
- **Risk:** If feature names change between training and inference, SHAP interpretation breaks

### Bridge 2: Model Interface Contract
- **Expected by:** ExplainabilityAnalyzer, TradeExceptionScoringService
- **Required methods:** `predict()`, `predict_proba()` 
- **Used by:** Optuna objective function during tuning
- **Implication:** Any model swap must preserve this sklearn-compatible interface

### Bridge 3: MLflow Tracking
- **Logged by:** TradeExceptionPredictor._evaluate()
- **Registry location:** `./mlflow_tracking/` directory
- **Contents:** Hyperparameters, test metrics, model artifact (xgboost binary)
- **Not logged:** Feature importance (could improve pipeline reproducibility)

### Bridge 4: Serialization Contracts
- **FeatureEngineering state:** Serialized as pkl (label_encoders dict, scaler object)
- **Model state:** Serialized as pkl (xgboost binary)
- **Python version risk:** Sklearn/XGBoost API changes can break old pkl files

---

## 📈 Suggested Questions to Explore

### Architecture Questions
1. **Why are interaction features multiplicative?** Why not additive or polynomial?
2. **How sensitive is model performance to the threshold choices in `high_doc_lag` (>3 days)?** Could we learn this from data?
3. **What happens if a trade has an unseen categorical value at inference?** The code has a fallback strategy — is it correct?

### Data Quality Questions
4. **Is `counterparty_exception_rate_30d` leakage?** Is it known at trade booking time or only at settlement?
5. **How much of the exception prediction is driven by operational flags (doc lag, amendments) vs. market flags (volatility, currency mismatch)?**
6. **Are there systematic differences in exception rates by day of week or month-end timing that could indicate data generation artifacts?**

### Model Behavior Questions
7. **What is the Optuna trial success rate? Are many trials pruned?** Indicates hyperparameter sensitivity.
8. **How does model performance degrade on out-of-distribution trades?** (e.g., new commodities, new counterparties)
9. **Is there interaction between `tier_x_month_end` and `counterparty_exception_rate_30d`?** Could suggest compound risk effects.

### Production Readiness Questions
10. **How frequently should the model be retrained?** Daily? Weekly? When new exception patterns emerge?
11. **Are SHAP explanations cached for audit trails, or recomputed per-request?** Production performance vs. interpretability trade-off.
12. **What is the latency budget for the Azure ML endpoint?** Does single-trade scoring with SHAP stay under SLA?

### Improvement Opportunities
13. **Feature importance stability:** Do top 5 drivers stay consistent across cross-validation folds?
14. **Calibration:** Are predicted probabilities well-calibrated? (0.7 predicted should ≈ 70% actual exceptions)
15. **Ensemble potential:** Would boosting + stacking improve over single XGBoost? Trade-off: interpretability.

---

## 🛠️ Key Implementation Notes

### Strengths
- ✅ **Clear separation of concerns** — Feature engineering, training, explanation, scoring are isolated
- ✅ **Reusable feature transformer** — Same logic for training and inference
- ✅ **Explainability built-in** — SHAP for every prediction, not an afterthought
- ✅ **MLflow integration** — Experiment tracking from day 1
- ✅ **Class imbalance handling** — scale_pos_weight computed and applied

### Potential Gaps
- ⚠️ **No feature importance logging to MLflow** — Missing experiment-to-experiment reproducibility signal
- ⚠️ **SHAP computation not cached** — Each inference request recomputes, potential latency issue
- ⚠️ **No drift detection** — No mechanism to alert when test-time data distribution shifts
- ⚠️ **Confidence thresholds hard-coded** — Business rules buried in score.py, hard to A/B test
- ⚠️ **No cross-validation** — Single train-test split, no k-fold assessment of generalization

### Production Readiness Checklist
- [ ] Serialize trained FeatureEngineering object (feature_engineer.pkl)
- [ ] Serialize trained model (xgboost_model.pkl) with version tag
- [ ] Define SLA for Azure ML endpoint latency (single-trade scoring + SHAP explanation)
- [ ] Set up monitoring for input data drift (monitor feature distributions)
- [ ] Define retraining trigger policy (e.g., retrain if test AUC drops below 0.82)
- [ ] Document feature meanings and thresholds (e.g., why high_doc_lag = >3 days?)
- [ ] Test backward compatibility with new categorical values at inference

---

## 📚 Codebase Statistics

| Component | Lines | Key Classes | Dependencies |
|-----------|-------|-------------|--------------|
| features.py | 250 | FeatureEngineering (1 class) | sklearn, pandas |
| train.py | 200 | TradeExceptionPredictor (1 class) | xgboost, optuna, mlflow |
| explain.py | 175 | ExplainabilityAnalyzer (1 class) | shap, matplotlib |
| score.py | 420 | TradeExceptionScoringService (1 class) | xgboost, shap, joblib |
| **Total** | **~1,000** | **4 main classes** | **~10 libraries** |

**Notebooks:**
- `01_eda.ipynb` — Data exploration and visualization
- `02_modeling.ipynb` — Full training pipeline (features → train → explain)

---

## 🔍 Community Detection (Conceptual)

If graphify were analyzing this codebase, it would detect these communities:

**Community 1: Feature Processing**
- `FeatureEngineering` ↔ `prepare_features()` ↔ sklearn encoders
- **Cohesion:** All focused on input feature transformation
- **Role:** Data preparation layer

**Community 2: Model Training**
- `TradeExceptionPredictor` ↔ Optuna ↔ XGBoost ↔ MLflow
- **Cohesion:** All focused on hyperparameter tuning and training
- **Role:** ML pipeline layer

**Community 3: Explainability**
- `ExplainabilityAnalyzer` ↔ SHAP ↔ matplotlib
- **Cohesion:** All focused on model interpretability
- **Role:** Explanation layer

**Community 4: Production Serving**
- `TradeExceptionScoringService` ↔ FeatureEngineering (inference mode) ↔ score.py endpoints
- **Cohesion:** All focused on real-time or batch inference
- **Role:** Serving layer

**Surprising Connection (Cross-Community):**
- `FeatureEngineering.transform()` is called by both `ExplainabilityAnalyzer` (during analysis) and `TradeExceptionScoringService` (during inference)
- This makes feature engineering a **bottleneck for consistency** — any bug in transform() breaks both explanation and serving

---

## 🎓 For New Contributors

Start here:
1. Read CLAUDE.md for project spec
2. Run `01_eda.ipynb` to understand data
3. Run `02_modeling.ipynb` to see full pipeline
4. Modify a feature in `features.py` and trace through to `explain.py` to see impact

Key files to avoid breaking:
- **features.py** — Used by training and inference; state must be serializable
- **train.py** — Hyperparameter ranges tuned for this dataset; changing them requires re-tuning
- **score.py** endpoints — Azure ML interface; signature changes break integration

---

**Generated Knowledge Map | April 28, 2026**
