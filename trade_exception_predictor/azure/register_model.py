"""
register_model.py - Register Trained Model to Azure ML Registry

Registers the trade exception model from local artifacts or MLflow to Azure ML.
Works with all trained models from:
  - train_updated.py (local)
  - train_azure_wrapper.py (Azure ML)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
import json
import joblib

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=str(env_path))


def load_azure_config():
    """Load Azure configuration from .env or environment."""
    
    config = {
        'subscription_id': os.getenv('subscription_id', '').strip().strip('"'),
        'resource_group': os.getenv('resource_group_name', '').strip().strip('"'),
        'workspace_name': os.getenv('workspace_name', '').strip().strip('"'),
    }
    
    # Validate
    for key, value in config.items():
        if not value:
            raise ValueError(f"Missing {key}. Set in .env file or environment.")
    
    return config


def register_model_from_local(
    model_path: str,
    feature_engineer_path: str,
    feature_names_path: str = None,
    model_name: str = 'trade-exception-predictor',
    model_version: str = '1.0',
    description: str = None
):
    """
    Register model from local artifacts to Azure ML.
    
    Parameters
    ----------
    model_path : str
        Path to xgboost_model.pkl
    feature_engineer_path : str
        Path to feature_engineer.pkl
    feature_names_path : str
        Path to feature_importance.csv (optional)
    model_name : str
        Model name in Azure ML registry
    model_version : str
        Model version
    description : str
        Model description
    
    Returns
    -------
    azure.ai.ml.entities.Model
        Registered model
    """
    
    print("\n" + "="*80)
    print("REGISTER MODEL FROM LOCAL ARTIFACTS")
    print("="*80)
    
    # Load configuration
    config = load_azure_config()
    
    print(f"\nConfiguration:")
    print(f"  Subscription ID: {config['subscription_id']}")
    print(f"  Resource Group: {config['resource_group']}")
    print(f"  Workspace: {config['workspace_name']}")
    
    # Authenticate
    print(f"\nAuthenticating...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        config['subscription_id'],
        config['resource_group'],
        config['workspace_name']
    )
    print(f"[OK] Authenticated to Azure ML")
    
    # Validate paths
    model_path = Path(model_path)
    feature_engineer_path = Path(feature_engineer_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feature_engineer_path.exists():
        raise FileNotFoundError(f"Feature engineer not found: {feature_engineer_path}")
    
    print(f"\nArtifact Paths:")
    print(f"  Model: {model_path}")
    print(f"  Feature Engineer: {feature_engineer_path}")
    
    # Load model to extract metadata
    print(f"\nExtracting model metadata...")
    try:
        model_obj = joblib.load(model_path)
        print(f"  Model type: {type(model_obj).__name__}")
        print(f"  Best iteration: {model_obj.best_iteration}")
    except Exception as e:
        print(f"  Warning: Could not extract metadata: {str(e)}")
    
    # Register to Azure ML
    print(f"\nRegistering model...")
    
    model_description = description or (
        "Trade exception prediction model trained with XGBoost and Optuna. "
        "Features: 33 engineered features from 25 raw trade columns. "
        "Explainability: SHAP values for every prediction."
    )
    
    model = Model(
        path=str(model_path),
        name=model_name,
        version=model_version,
        type='custom_model',
        description=model_description,
        tags={
            'project': 'trade_exception_predictor',
            'framework': 'xgboost',
            'optimization': 'optuna',
            'explainability': 'shap',
            'cloud': 'azure_ml'
        },
        properties={
            'input_columns': '25',
            'output_features': '33',
            'target': 'is_exception',
            'class_imbalance_weight': '2.44'
        }
    )
    
    registered_model = ml_client.models.create_or_update(model)
    
    print(f"\n" + "="*80)
    print(f"[OK] MODEL REGISTERED")
    print(f"="*80)
    print(f"\nModel Details:")
    print(f"  Name: {registered_model.name}")
    print(f"  Version: {registered_model.version}")
    print(f"  ID: {registered_model.id}")
    print(f"  Type: {registered_model.type}")
    
    return registered_model


def register_model_from_mlflow(
    mlflow_run_id: str,
    model_name: str = 'trade-exception-predictor',
    model_version: str = '1.0'
):
    """
    Register model from MLflow to Azure ML.
    
    Parameters
    ----------
    mlflow_run_id : str
        MLflow run ID
    model_name : str
        Model name in Azure ML registry
    model_version : str
        Model version
    
    Returns
    -------
    azure.ai.ml.entities.Model
        Registered model
    """
    
    print("\n" + "="*80)
    print("REGISTER MODEL FROM MLFLOW")
    print("="*80)
    
    # Load configuration
    config = load_azure_config()
    
    print(f"\nConfiguration:")
    print(f"  Subscription ID: {config['subscription_id']}")
    print(f"  Resource Group: {config['resource_group']}")
    print(f"  Workspace: {config['workspace_name']}")
    print(f"  MLflow Run ID: {mlflow_run_id}")
    
    # Authenticate
    print(f"\nAuthenticating...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        config['subscription_id'],
        config['resource_group'],
        config['workspace_name']
    )
    print(f"[OK] Authenticated to Azure ML")
    
    # Load MLflow run metadata
    print(f"\nFetching MLflow run metadata...")
    import mlflow
    mlflow_client = mlflow.tracking.MlflowClient()
    run = mlflow_client.get_run(mlflow_run_id)
    
    print(f"  Status: {run.info.status}")
    print(f"  Start time: {run.info.start_time}")
    
    # Extract metrics
    metrics = run.data.metrics
    test_auc = metrics.get('test_auc', 'N/A')
    test_precision = metrics.get('test_precision', 'N/A')
    test_recall = metrics.get('test_recall', 'N/A')
    test_f1 = metrics.get('test_f1', 'N/A')
    
    print(f"\nModel Metrics:")
    print(f"  AUC-ROC: {test_auc}")
    print(f"  Precision: {test_precision}")
    print(f"  Recall: {test_recall}")
    print(f"  F1-Score: {test_f1}")
    
    # Register to Azure ML
    print(f"\nRegistering model from MLflow...")
    
    model = Model(
        path=f'runs:/{mlflow_run_id}/xgboost_model',
        name=model_name,
        version=model_version,
        type='xgboost',
        description=(
            "Trade exception prediction model trained with XGBoost and Optuna. "
            "Features: 33 engineered features from 25 raw trade columns. "
            "Explainability: SHAP values for every prediction."
        ),
        tags={
            'project': 'trade_exception_predictor',
            'framework': 'xgboost',
            'optimization': 'optuna',
            'explainability': 'shap',
            'mlflow_run_id': mlflow_run_id
        },
        properties={
            'auc': str(test_auc),
            'precision': str(test_precision),
            'recall': str(test_recall),
            'f1_score': str(test_f1)
        }
    )
    
    registered_model = ml_client.models.create_or_update(model)
    
    print(f"\n" + "="*80)
    print(f"[OK] MODEL REGISTERED")
    print(f"="*80)
    print(f"\nModel Details:")
    print(f"  Name: {registered_model.name}")
    print(f"  Version: {registered_model.version}")
    print(f"  ID: {registered_model.id}")
    
    return registered_model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Register model to Azure ML')
    parser.add_argument(
        '--source',
        choices=['local', 'mlflow'],
        required=True,
        help='Model source'
    )
    parser.add_argument(
        '--model_path',
        help='Path to xgboost_model.pkl (for local source)'
    )
    parser.add_argument(
        '--feature_engineer_path',
        help='Path to feature_engineer.pkl (for local source)'
    )
    parser.add_argument(
        '--mlflow_run_id',
        help='MLflow run ID (for mlflow source)'
    )
    parser.add_argument(
        '--model_name',
        default='trade-exception-predictor',
        help='Model name in Azure ML'
    )
    parser.add_argument(
        '--model_version',
        default='1.0',
        help='Model version'
    )
    
    args = parser.parse_args()
    
    try:
        if args.source == 'local':
            if not args.model_path or not args.feature_engineer_path:
                print("Error: --model_path and --feature_engineer_path required for local source")
                sys.exit(1)
            
            register_model_from_local(
                args.model_path,
                args.feature_engineer_path,
                model_name=args.model_name,
                model_version=args.model_version
            )
        
        elif args.source == 'mlflow':
            if not args.mlflow_run_id:
                print("Error: --mlflow_run_id required for mlflow source")
                sys.exit(1)
            
            register_model_from_mlflow(
                args.mlflow_run_id,
                model_name=args.model_name,
                model_version=args.model_version
            )
        
        print(f"\n[OK] Model registration complete!")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n[FAIL] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)