"""Submit training job to Azure ML."""
import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from pathlib import Path

# Load environment variables from .env file (parent directory)
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env'))
load_dotenv(dotenv_path=env_path)


def submit_training_job(
    subscription_id: str = None,
    resource_group: str = None,
    workspace_name: str = None,
    compute_target: str = 'cpu-cluster'
):
    """Submit training job to Azure ML."""

    # Use environment variables if not provided
    subscription_id = subscription_id or os.getenv('subscription_id')
    resource_group = resource_group or os.getenv('resource_group_name')
    workspace_name = workspace_name or os.getenv('workspace_name')

    # Clean up whitespace and quotes from .env file
    subscription_id = subscription_id.strip().strip('"')
    resource_group = resource_group.strip().strip('"')
    workspace_name = workspace_name.strip().strip('"')

    print(f"Using Azure credentials:")
    print(f"  Subscription: {subscription_id}")
    print(f"  Resource Group: {resource_group}")
    print(f"  Workspace: {workspace_name}")

    # Authenticate
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential, subscription_id, resource_group, workspace_name
    )

    # Create environment from conda spec
    # Get absolute path to environment.yml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(script_dir, '../environment.yml')
    env_file = os.path.abspath(env_file)

    print(f"Loading environment from: {env_file}")

    env = Environment(
        name='trade-exception-env',
        description='Environment for trade exception prediction',
        conda_file=env_file
    )

    # Get absolute paths
    src_path = os.path.abspath(os.path.join(script_dir, '../src'))
    data_path = os.path.abspath(os.path.join(script_dir, '../data/trades_synthetic.csv'))

    print(f"Code path: {src_path}")
    print(f"Data path: {data_path}")

    # Define training command
    training_job = command(
        code=src_path,
        command='python train.py',
        inputs={
            'training_data': Input(type='uri_file', path=data_path)
        },
        outputs={
            'model': Output(type='uri_folder')
        },
        environment=env,
        compute=compute_target,
        display_name='trade_exception_training',
        description='Train XGBoost model for trade exception prediction',
        tags={'project': 'trade_exception_predictor', 'model': 'xgboost'}
    )

    # Submit job
    job = ml_client.create_or_update(training_job)
    print(f"Job submitted: {job.name}")
    print(f"Job URL: {job.studio_url}")

    return job


if __name__ == '__main__':
    import sys

    # Load from .env if no args provided
    if len(sys.argv) < 2:
        print("Loading credentials from .env file...")
        job = submit_training_job()
    else:
        # Allow override via command line
        subscription_id = sys.argv[1]
        resource_group = sys.argv[2]
        workspace_name = sys.argv[3]
        compute_target = sys.argv[4] if len(sys.argv) > 4 else 'cpu-cluster'

        job = submit_training_job(
            subscription_id, resource_group, workspace_name, compute_target
        )
