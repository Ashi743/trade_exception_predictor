"""
deploy_endpoint.py - Deploy Model to Azure ML Online Endpoint

Deploys the trade exception model as an online endpoint with:
  - SHAP explanations
  - Batch scoring support
  - Real-time inference
  - Auto-scaling
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
    Environment
)
from azure.identity import DefaultAzureCredential
import yaml

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=str(env_path))


def load_azure_config():
    """Load Azure configuration from .env or environment."""
    
    config = {
        'subscription_id': os.getenv('subscription_id', ''),
        'resource_group': os.getenv('resource_group_name', ''),
        'workspace_name': os.getenv('workspace_name', ''),
    }
    
    # Validate
    for key, value in config.items():
        if not value:
            raise ValueError(f"Missing {key}. Set in .env file or environment.")
    
    return config


def deploy_endpoint(
    model_name: str = 'trade-exception-predictor',
    model_version: str = '1.0',
    endpoint_name: str = 'trade-exception-endpoint',
    deployment_name: str = 'xgboost-deployment',
    instance_type: str = 'Standard_DS2_v2',
    instance_count: int = 2,
    min_instances: int = 1,
    max_instances: int = 4
):
    """
    Deploy model to Azure ML online endpoint.
    
    Parameters
    ----------
    model_name : str
        Model name in Azure ML registry
    model_version : str
        Model version
    endpoint_name : str
        Online endpoint name
    deployment_name : str
        Deployment name
    instance_type : str
        Compute instance type
    instance_count : int
        Initial instance count
    min_instances : int
        Minimum auto-scale instances
    max_instances : int
        Maximum auto-scale instances
    
    Returns
    -------
    azure.ai.ml.entities.ManagedOnlineEndpoint
        Deployed endpoint
    """
    
    print("\n" + "="*80)
    print("DEPLOY MODEL TO AZURE ML ONLINE ENDPOINT")
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
    
    # Check if model exists
    print(f"\nValidating model...")
    try:
        model = ml_client.models.get(model_name, model_version)
        print(f"[OK] Model found: {model.name} (v{model.version})")
    except Exception as e:
        print(f"[FAIL] Model not found: {str(e)}")
        print(f"  Register model first using register_model.py")
        raise
    
    # Create endpoint
    print(f"\nCreating endpoint...")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description='Trade exception prediction endpoint with SHAP explanations',
        auth_mode='key',
        tags={
            'project': 'trade_exception_predictor',
            'model': 'xgboost',
            'explainability': 'shap'
        }
    )
    
    try:
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"[OK] Endpoint created/updated: {endpoint.name}")
        print(f"  Endpoint URI: {endpoint.scoring_uri}")
    except Exception as e:
        print(f"[FAIL] Failed to create endpoint: {str(e)}")
        raise
    
    # Create deployment with scoring script
    print(f"\nCreating deployment...")
    
    # Resolve paths
    script_dir = Path(__file__).parent.parent
    score_script = script_dir / 'src' / 'score.py'
    env_file = script_dir / 'environment.yml'

    if not score_script.exists():
        raise FileNotFoundError(f"Scoring script not found: {score_script}")
    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    print(f"  Scoring script: {score_script}")
    print(f"  Environment: {env_file}")
    
    # Create environment
    env = Environment(
        conda_file=str(env_file),
        name='trade-exception-deploy-env',
        description='Environment for trade exception endpoint'
    )
    
    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model.id,
        code_configuration=CodeConfiguration(
                code=str(script_dir / 'src'),
                scoring_script='score.py'
                ),
                environment=env,
                instance_type=instance_type,
                instance_count=instance_count,
                properties={
                            'min_instances': str(min_instances),
                            'max_instances': str(max_instances)
                        },
                        description='XGBoost deployment with SHAP explanations',
                        tags={
                            'model_name': model_name,
                            'model_version': model_version
        }
    )
    
    try:
        deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
        print(f"[OK] Deployment created: {deployment.name}")
    except Exception as e:
        print(f"[FAIL] Failed to create deployment: {str(e)}")
        raise
    
    # Set traffic to new deployment
    print(f"\nConfiguring traffic...")
    endpoint.traffic = {deployment_name: 100}
    
    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"[OK] Traffic configured (100% to {deployment_name})")
    except Exception as e:
        print(f"[WARNING]  Warning: Could not configure traffic: {str(e)}")
    
    # Get endpoint details
    print(f"\n" + "="*80)
    print(f"[OK] DEPLOYMENT COMPLETE")
    print(f"="*80)
    
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    
    print(f"\nEndpoint Details:")
    print(f"  Name: {endpoint.name}")
    print(f"  Status: {endpoint.provisioning_state}")
    print(f"  Scoring URI: {endpoint.scoring_uri}")
    print(f"  Deployment: {deployment_name}")
    
    # Get keys
    print(f"\nAuthentication Keys:")
    keys = ml_client.online_endpoints.get_keys(endpoint_name)
    print(f"  Primary key: {keys.primary_key[:20]}...")
    print(f"  Secondary key: {keys.secondary_key[:20]}...")
    
    print(f"\nUsage Example:")
    print(f"""
  import requests
  import json
  
  url = "{endpoint.scoring_uri}"
  headers = {{"Authorization": f"Bearer <key>"}}
  
  # Single trade
  data = {{
    "trade_id": "TR_001",
    "commodity_type": "Soybean",
    "counterparty_tier": "Tier1",
    ... (all 25 columns)
  }}
  
  response = requests.post(url, json=data, headers=headers)
  result = response.json()
  
  # Result contains:
  # - prediction (0=clean, 1=exception)
  # - exception_probability
  # - confidence (high/medium/low)
  # - top_drivers (SHAP explanations)
  # - recommendation (operational action)
    """)
    
    return endpoint


def delete_endpoint(endpoint_name: str):
    """Delete online endpoint."""
    
    print(f"\nDeleting endpoint: {endpoint_name}")
    
    config = load_azure_config()
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        config['subscription_id'],
        config['resource_group'],
        config['workspace_name']
    )
    
    ml_client.online_endpoints.begin_delete(endpoint_name).result()
    print(f"[OK] Endpoint deleted")


def test_endpoint(endpoint_name: str):
    """Test endpoint with sample data."""
    
    print(f"\nTesting endpoint: {endpoint_name}")
    
    config = load_azure_config()
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        config['subscription_id'],
        config['resource_group'],
        config['workspace_name']
    )
    
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    
    # Sample trade
    test_data = {
        "trade_id": "TR_TEST_001",
        "trade_date": "2024-01-15",
        "settlement_date": "2024-01-20",
        "days_to_settlement": 5,
        "commodity_type": "Soybean",
        "instrument_type": "Spot",
        "delivery_location": "Port_A",
        "counterparty_id": "CP_001",
        "counterparty_tier": "Tier1",
        "counterparty_region": "North_America",
        "notional_usd": 100000.0,
        "quantity_mt": 100.0,
        "price_per_mt": 500.0,
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
    }
    
    import requests
    import json
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        endpoint.scoring_uri,
        json=test_data,
        headers=headers
    )
    
    print(f"\nTest Result:")
    print(json.dumps(response.json(), indent=2))


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy model to Azure ML endpoint')
    parser.add_argument(
        '--action',
        choices=['deploy', 'delete', 'test'],
        default='deploy',
        help='Action to perform'
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
    parser.add_argument(
        '--endpoint_name',
        default='trade-exception-endpoint',
        help='Endpoint name'
    )
    parser.add_argument(
        '--deployment_name',
        default='xgboost-deployment',
        help='Deployment name'
    )
    parser.add_argument(
        '--instance_type',
        default='Standard_DS2_v2',
        help='Compute instance type'
    )
    parser.add_argument(
        '--instance_count',
        type=int,
        default=2,
        help='Initial instance count'
    )
    parser.add_argument(
        '--min_instances',
        type=int,
        default=1,
        help='Minimum auto-scale instances'
    )
    parser.add_argument(
        '--max_instances',
        type=int,
        default=4,
        help='Maximum auto-scale instances'
    )
    
    args = parser.parse_args()
    
    try:
        if args.action == 'deploy':
            deploy_endpoint(
                model_name=args.model_name,
                model_version=args.model_version,
                endpoint_name=args.endpoint_name,
                deployment_name=args.deployment_name,
                instance_type=args.instance_type,
                instance_count=args.instance_count,
                min_instances=args.min_instances,
                max_instances=args.max_instances
            )
        
        elif args.action == 'delete':
            delete_endpoint(args.endpoint_name)
        
        elif args.action == 'test':
            test_endpoint(args.endpoint_name)
        
        print(f"\n[OK] Action completed successfully!")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n[FAIL] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)