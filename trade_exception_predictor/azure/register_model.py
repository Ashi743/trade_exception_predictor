"""Register trained model to Azure ML registry."""
import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from pathlib import Path
import mlflow

# Load environment variables from .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env'))
load_dotenv(dotenv_path=env_path)


def register_model_from_mlflow(
    subscription_id: str = None,
    resource_group: str = None,
    workspace_name: str = None,
    mlflow_run_id: str = None,
    model_name: str = 'trade-exception-model',
    model_version: str = '1.0'
):
    """Register model from MLflow to Azure ML."""

    # Use environment variables if not provided
    subscription_id = subscription_id or os.getenv('subscription_id')
    resource_group = resource_group or os.getenv('resource_group_name')
    workspace_name = workspace_name or os.getenv('workspace_name')

    # Clean up whitespace and quotes
    subscription_id = subscription_id.strip().strip('"')
    resource_group = resource_group.strip().strip('"')
    workspace_name = workspace_name.strip().strip('"')

    print(f"Registering model with:")
    print(f"  Subscription: {subscription_id}")
    print(f"  Resource Group: {resource_group}")
    print(f"  Workspace: {workspace_name}")
    print(f"  MLflow Run ID: {mlflow_run_id}")

    # Authenticate
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential, subscription_id, resource_group, workspace_name
    )

    # Load model from MLflow
    mlflow_client = mlflow.tracking.MlflowClient()
    run = mlflow_client.get_run(mlflow_run_id)

    # Register to Azure ML
    model = Model(
        path=f'runs:/{mlflow_run_id}/model',
        name=model_name,
        version=model_version,
        type='xgboost',
        description='Trade exception prediction model trained with XGBoost and Optuna',
        tags={
            'project': 'trade_exception_predictor',
            'framework': 'xgboost',
            'mlflow_run_id': mlflow_run_id
        },
        properties={
            'auc': str(run.data.metrics.get('test_auc', 'N/A')),
            'f1_score': str(run.data.metrics.get('test_f1', 'N/A'))
        }
    )

    registered_model = ml_client.models.create_or_update(model)
    print(f"✓ Model registered: {registered_model.name} (v{registered_model.version})")

    return registered_model


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Loading credentials from .env file...")
        mlflow_run_id = input("Enter MLflow Run ID: ")
        register_model_from_mlflow(mlflow_run_id=mlflow_run_id)
    else:
        # Allow override via command line
        subscription_id = sys.argv[1] if len(sys.argv) > 1 else None
        resource_group = sys.argv[2] if len(sys.argv) > 2 else None
        workspace_name = sys.argv[3] if len(sys.argv) > 3 else None
        mlflow_run_id = sys.argv[4] if len(sys.argv) > 4 else None
        model_name = sys.argv[5] if len(sys.argv) > 5 else 'trade-exception-model'
        model_version = sys.argv[6] if len(sys.argv) > 6 else '1.0'

        register_model_from_mlflow(
            subscription_id, resource_group, workspace_name,
            mlflow_run_id, model_name, model_version
        )
