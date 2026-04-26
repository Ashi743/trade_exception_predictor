"""Deploy model to Azure ML endpoint."""
import os
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
from pathlib import Path

# Load environment variables from .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env'))
load_dotenv(dotenv_path=env_path)


def deploy_model_endpoint(
    subscription_id: str = None,
    resource_group: str = None,
    workspace_name: str = None,
    model_name: str = None,
    model_version: str = None,
    endpoint_name: str = 'trade-exception-endpoint',
    deployment_name: str = 'xgboost-deploy'
):
    """Deploy model to managed online endpoint."""

    # Use environment variables if not provided
    subscription_id = subscription_id or os.getenv('subscription_id')
    resource_group = resource_group or os.getenv('resource_group_name')
    workspace_name = workspace_name or os.getenv('workspace_name')

    # Clean up whitespace and quotes
    subscription_id = subscription_id.strip().strip('"')
    resource_group = resource_group.strip().strip('"')
    workspace_name = workspace_name.strip().strip('"')

    print(f"Deploying with:")
    print(f"  Subscription: {subscription_id}")
    print(f"  Resource Group: {resource_group}")
    print(f"  Workspace: {workspace_name}")
    print(f"  Model: {model_name} (v{model_version})")

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.abspath(os.path.join(script_dir, '../environment.yml'))
    src_path = os.path.abspath(os.path.join(script_dir, '../src'))

    print(f"  Environment: {env_file}")
    print(f"  Code: {src_path}")

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential, subscription_id, resource_group, workspace_name
    )

    # Create endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description='Trade exception prediction endpoint',
        auth_mode='key',
        tags={'project': 'trade_exception_predictor'}
    )

    print(f'\nCreating endpoint {endpoint_name}...')
    endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f'✓ Endpoint created: {endpoint_result.scoring_uri}')

    # Get model reference
    model = ml_client.models.get(name=model_name, version=model_version)

    # Create environment
    env = Environment(
        conda_file=env_file,
        name='trade-exception-env',
        description='Environment for trade exception prediction'
    )

    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        code_configuration=CodeConfiguration(
            code=src_path,
            scoring_script='score.py'
        ),
        environment=env,
        instance_type='Standard_DS2_v2',
        instance_count=1,
        tags={'deployment': 'xgboost'}
    )

    print(f'Creating deployment {deployment_name}...')
    deployment_result = ml_client.online_deployments.begin_create_or_update(
        deployment
    ).result()

    # Update traffic to new deployment
    endpoint_result.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint_result).result()

    print(f'✓ Deployment completed!')
    print(f'Endpoint: {endpoint_result.scoring_uri}')
    print(f'Deployment: {deployment_result.name}')

    return endpoint_result, deployment_result


def test_endpoint(
    subscription_id: str = None,
    resource_group: str = None,
    workspace_name: str = None,
    endpoint_name: str = None
):
    """Test deployed endpoint."""

    # Use environment variables if not provided
    subscription_id = subscription_id or os.getenv('subscription_id')
    resource_group = resource_group or os.getenv('resource_group_name')
    workspace_name = workspace_name or os.getenv('workspace_name')

    # Clean up whitespace and quotes
    subscription_id = subscription_id.strip().strip('"')
    resource_group = resource_group.strip().strip('"')
    workspace_name = workspace_name.strip().strip('"')

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential, subscription_id, resource_group, workspace_name
    )

    # Sample input
    sample_input = {
        'counterparty': 'Bank_A',
        'instrument_type': 'FX',
        'notional_amount': 1000000,
        'trade_price': 1.0850,
        'market_volatility': 0.12,
        'counterparty_risk_score': 0.15,
        'execution_speed_ms': 45,
        'price_deviation_pct': 0.05,
        'trade_size_percentile': 65
    }

    # Score
    result = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        request_file=sample_input
    )

    print(f'✓ Prediction result: {result}')
    return result


if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("Loading credentials from .env file...")
        model_name = input("Enter Model Name: ") or 'trade-exception-model'
        model_version = input("Enter Model Version: ") or '1.0'
        endpoint_name = input("Enter Endpoint Name (default: trade-exception-endpoint): ") or 'trade-exception-endpoint'
        deploy_model_endpoint(
            model_name=model_name,
            model_version=model_version,
            endpoint_name=endpoint_name
        )
    else:
        # Allow override via command line
        subscription_id = sys.argv[1] if len(sys.argv) > 1 else None
        resource_group = sys.argv[2] if len(sys.argv) > 2 else None
        workspace_name = sys.argv[3] if len(sys.argv) > 3 else None
        model_name = sys.argv[4] if len(sys.argv) > 4 else None
        model_version = sys.argv[5] if len(sys.argv) > 5 else None
        endpoint_name = sys.argv[6] if len(sys.argv) > 6 else 'trade-exception-endpoint'

        deploy_model_endpoint(
            subscription_id, resource_group, workspace_name,
            model_name, model_version, endpoint_name
        )
