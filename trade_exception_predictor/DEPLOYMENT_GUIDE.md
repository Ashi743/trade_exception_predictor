# Trade Exception Predictor - Azure ML Deployment Guide

**Status:** Ready for deployment  
**Configuration:** ✓ Azure credentials authenticated  
**Account:** amaranthony0786@gmail.com  
**Subscription:** Azure subscription 1 (6b2e8e6e-3006-43df-b00a-3499ebb11605)

---

## Pre-Deployment Checklist

Before deploying to Azure ML, ensure:

- [x] Logged in to Azure (`az login` successful)
- [x] `.env` file configured with:
  - Subscription ID: `6b2e8e6e-3006-43df-b00a-3499ebb11605`
  - Resource Group: `azurexgb-rg`
  - Workspace: `trade-exception-ws`
- [x] Model trained locally (produces `xgboost_model.pkl` and `feature_engineer.pkl`)
- [x] Scoring script ready (`src/score.py`)
- [x] Environment spec available (`environment.yml`)

---

## Deployment Workflow

### Step 1: Train and Save Model Locally

**Command:**
```bash
python notebooks/02_modeling.ipynb
```

**Produces:**
- `xgboost_model.pkl` (trained XGBoost binary classifier)
- `feature_engineer.pkl` (fitted FeatureEngineering transformer)
- MLflow runs logged to `./mlflow_tracking/`

**Where to save artifacts:**
Create `outputs/` directory in project root:
```bash
mkdir -p outputs
cp xgboost_model.pkl outputs/
cp feature_engineer.pkl outputs/
```

### Step 2: Register Model to Azure ML

**Command:**
```bash
python azure/register_model.py \
  --model_name trade-exception-predictor \
  --model_version 1.0 \
  --model_path outputs/xgboost_model.pkl \
  --feature_engineer_path outputs/feature_engineer.pkl
```

**What it does:**
- Authenticates with Azure ML (uses `.env` config)
- Uploads model artifacts to Azure ML registry
- Creates model version `trade-exception-predictor:1.0`
- Logs model metadata and tags

**Success indicator:**
```
✓ REGISTRATION COMPLETE
Model: trade-exception-predictor (v1.0)
Registered to Azure ML workspace: trade-exception-ws
```

### Step 3: Deploy to Online Endpoint

**Command:**
```bash
python azure/deploy_endpoint.py \
  --action deploy \
  --model_name trade-exception-predictor \
  --model_version 1.0 \
  --endpoint_name trade-exception-endpoint \
  --deployment_name xgboost-deployment \
  --instance_type Standard_DS2_v2 \
  --instance_count 2 \
  --min_instances 1 \
  --max_instances 4
```

**What it does:**
1. Creates Azure ML managed online endpoint (DNS resolvable)
2. Deploys model with scoring script (`score.py`)
3. Configures auto-scaling (1-4 instances)
4. Sets 100% traffic to new deployment
5. Returns endpoint URI and authentication keys

**Success indicator:**
```
✓ DEPLOYMENT COMPLETE
Name: trade-exception-endpoint
Status: Healthy
Scoring URI: https://trade-exception-endpoint.westus2.inference.ml.azure.com/score
Primary key: ...
```

**Estimated time:** 5-10 minutes

**Compute cost:** ~$0.50/hour for Standard_DS2_v2 x2 instances

---

## Usage: Scoring Trades

### Real-Time Single Trade Scoring

**Request:**
```bash
curl -X POST https://trade-exception-endpoint.westus2.inference.ml.azure.com/score \
  -H "Authorization: Bearer <primary-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "trade_id": "TR_001",
    "trade_date": "2024-01-15",
    "settlement_date": "2024-01-20",
    "days_to_settlement": 5,
    "commodity_type": "Soybean",
    "instrument_type": "Spot",
    "delivery_location": "Port_A",
    "counterparty_id": "CP_001",
    "counterparty_tier": "Tier1",
    "counterparty_region": "North_America",
    "notional_usd": 100000,
    "quantity_mt": 100,
    "price_per_mt": 500,
    "settlement_currency": "USD",
    "is_month_end": 0,
    "is_quarter_end": 0,
    "day_of_week": "Monday",
    "counterparty_exception_rate_30d": 0.25,
    "same_commodity_breaks_7d": 1,
    "price_volatility_flag": 0,
    "cross_border_flag": 0,
    "currency_mismatch_flag": 0,
    "documentation_lag_days": 2,
    "amendment_count": 0
  }'
```

**Response:**
```json
{
  "status": "success",
  "trade_id": "TR_001",
  "prediction": 0,
  "prediction_label": "CLEAN",
  "exception_probability": 0.1234,
  "clean_probability": 0.8766,
  "confidence": "high",
  "top_drivers": [
    {
      "feature": "documentation_lag_days",
      "shap_value": -0.0521,
      "direction": "decreases_exception_risk"
    },
    {
      "feature": "price_volatility_flag",
      "shap_value": 0.0089,
      "direction": "increases_exception_risk"
    }
  ],
  "recommendation": "✓ VERY LOW RISK: 12.34% exception probability. Trade is clean.",
  "base_rate": 0.2910
}
```

### Python Client

```python
import requests
import json

endpoint_url = "https://trade-exception-endpoint.westus2.inference.ml.azure.com/score"
primary_key = "YOUR_PRIMARY_KEY"

headers = {
    "Authorization": f"Bearer {primary_key}",
    "Content-Type": "application/json"
}

trade = {
    "trade_id": "TR_001",
    "trade_date": "2024-01-15",
    # ... (all 25 columns)
}

response = requests.post(endpoint_url, json=trade, headers=headers)
result = response.json()

print(f"Prediction: {result['prediction_label']}")
print(f"Exception probability: {result['exception_probability']:.1%}")
print(f"Top driver: {result['top_drivers'][0]['feature']}")
```

### Batch Scoring

Send array of trades:

```python
trades = [
    {"trade_id": "TR_001", ...},
    {"trade_id": "TR_002", ...},
    {"trade_id": "TR_003", ...}
]

response = requests.post(endpoint_url, json=trades, headers=headers)
results = response.json()  # List of predictions
```

---

## Monitoring & Maintenance

### Check Endpoint Status

```bash
python -c "
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ml_client = MLClient(
    DefaultAzureCredential(),
    os.getenv('subscription_id').strip('\"'),
    os.getenv('resource_group_name').strip('\"'),
    os.getenv('workspace_name').strip('\"')
)

endpoint = ml_client.online_endpoints.get('trade-exception-endpoint')
print(f'Status: {endpoint.provisioning_state}')
print(f'URI: {endpoint.scoring_uri}')

deployment = ml_client.online_deployments.get(
    'trade-exception-endpoint',
    'xgboost-deployment'
)
print(f'Deployment status: {deployment.provisioning_state}')
print(f'Instance count: {deployment.instance_count}')
"
```

### View Logs

```bash
python azure/deploy_endpoint.py --action logs
```

### Scale Deployment

```bash
python -c "
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os

ml_client = MLClient(
    DefaultAzureCredential(),
    os.getenv('subscription_id').strip('\"'),
    os.getenv('resource_group_name').strip('\"'),
    os.getenv('workspace_name').strip('\"')
)

deployment = ml_client.online_deployments.get(
    'trade-exception-endpoint',
    'xgboost-deployment'
)

# Scale to 3 instances
deployment.instance_count = 3
ml_client.online_deployments.begin_create_or_update(deployment).result()
print('Scaled to 3 instances')
"
```

### Test Endpoint

```bash
python azure/deploy_endpoint.py --action test --endpoint_name trade-exception-endpoint
```

---

## Troubleshooting

### Issue: "Model not found" during deployment

**Cause:** Model not registered to Azure ML  
**Fix:** Run Step 2 (register_model.py) first

```bash
python azure/register_model.py \
  --model_path outputs/xgboost_model.pkl \
  --feature_engineer_path outputs/feature_engineer.pkl
```

### Issue: Scoring returns 400 error

**Cause:** Missing or malformed input data  
**Check:**
- All 25 required columns present
- Data types match expected (strings vs numbers)
- No null values in required fields

```python
# Validate input
required_cols = [
    'trade_id', 'trade_date', 'settlement_date', 'days_to_settlement',
    'commodity_type', 'instrument_type', 'delivery_location',
    'counterparty_id', 'counterparty_tier', 'counterparty_region',
    'notional_usd', 'quantity_mt', 'price_per_mt', 'settlement_currency',
    'is_month_end', 'is_quarter_end', 'day_of_week',
    'counterparty_exception_rate_30d', 'same_commodity_breaks_7d',
    'price_volatility_flag', 'cross_border_flag', 'currency_mismatch_flag',
    'documentation_lag_days', 'amendment_count'
]

for col in required_cols:
    if col not in trade_data:
        raise ValueError(f"Missing required column: {col}")
```

### Issue: Slow scoring latency (>1s per trade)

**Cause:** Small instance type or high traffic  
**Fix:** Scale up deployment

```bash
python azure/deploy_endpoint.py --action scale --min_instances 2 --max_instances 8
```

### Issue: "Unseen categorical value" during scoring

**Cause:** New commodity type or counterparty not in training data  
**Fix:** Feature engineering handles this gracefully:
- Unseen label-encoded values → mapped to first known class
- Unseen one-hot categories → dropped (zero for all dummies)

This is expected behavior; monitor for drift.

---

## Cost Estimation

**Azure ML Online Endpoint costs:**

| Component | Cost | Notes |
|-----------|------|-------|
| Managed Endpoint | Free | No charge for endpoint infrastructure |
| Compute | $0.50/hour per Standard_DS2_v2 instance | 2 instances = $1/hour |
| Storage | Minimal | Models <1GB total |
| **Monthly (with 2 instances 24/7)** | ~$360 | Eligible for Azure credits |

**Ways to reduce cost:**
- Use smaller instance type (Standard_D2_v2 = $0.25/hour)
- Set min_instances=0 and scale up on demand
- Use batch inference instead of real-time (Batch Endpoints, cheaper)

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Azure ML

on:
  push:
    branches: [main]
    paths:
      - 'src/score.py'
      - 'outputs/xgboost_model.pkl'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Deploy to Azure ML
        env:
          subscription_id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
          resource_group_name: ${{ secrets.AZURE_RESOURCE_GROUP }}
          workspace_name: ${{ secrets.AZURE_WORKSPACE }}
        run: |
          python azure/deploy_endpoint.py \
            --action deploy \
            --model_name trade-exception-predictor \
            --model_version 1.0
```

---

## Post-Deployment Tasks

1. **Monitor prediction performance** — Track metrics in Azure ML Studio
2. **Set up alerts** — Alert on high exception rates or errors
3. **Test failover** — Verify health checks and auto-recovery
4. **Document integration** — Update client applications with endpoint URI
5. **Schedule retraining** — Monthly or quarterly model updates

---

## Rollback & Cleanup

### Delete Endpoint (if needed)

```bash
python azure/deploy_endpoint.py --action delete --endpoint_name trade-exception-endpoint
```

This removes:
- Online endpoint and all deployments
- Compute resources and associated costs
- BUT preserves registered model in registry

### Restore from Previous Version

```bash
python azure/deploy_endpoint.py \
  --action deploy \
  --model_version 0.9 \
  --deployment_name xgboost-deployment-v0.9
```

Then re-route traffic:
```bash
python -c "
ml_client.online_endpoints.update_traffic(
    endpoint_name='trade-exception-endpoint',
    traffic={'xgboost-deployment-v0.9': 100}
)
"
```

---

## Next Steps

1. **Train model locally** → Run `02_modeling.ipynb`
2. **Save artifacts** → Move `.pkl` files to `outputs/`
3. **Register to Azure** → Run `register_model.py`
4. **Deploy endpoint** → Run `deploy_endpoint.py --action deploy`
5. **Test endpoint** → Run `deploy_endpoint.py --action test`
6. **Monitor in Azure ML Studio** → Check metrics and logs

**Timeline:** ~15 minutes for full deployment

**Questions?** Check Azure ML documentation or examine logs with:
```bash
python -c "
from azure.ai.ml import MLClient
ml_client = MLClient(...)
logs = ml_client.online_deployments.get_logs(
    'trade-exception-endpoint',
    'xgboost-deployment'
)
"
```

---

**Last Updated:** April 28, 2026
