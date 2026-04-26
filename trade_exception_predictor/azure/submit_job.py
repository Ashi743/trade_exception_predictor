"""
submit_job.py - Submit Training Job to Azure ML

Submits the complete trade exception predictor training pipeline to Azure ML.
Uses all updated modules:
  - features_updated.py
  - train_updated.py
  - explain_updated.py
  - score_updated.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import json

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


def submit_training_job(
    compute_target: str = 'cpu-cluster',
    n_trials: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    job_name: str = 'trade_exception_training'
):
    """
    Submit training job to Azure ML.
    
    Parameters
    ----------
    compute_target : str
        Azure ML compute cluster name
    n_trials : int
        Number of Optuna trials
    test_size : float
        Train/test split ratio
    random_state : int
        Random seed for reproducibility
    job_name : str
        Name for the job
    
    Returns
    -------
    azure.ai.ml.entities.Job
        Submitted job object
    """
    
    print("\n" + "="*80)
    print("AZURE ML JOB SUBMISSION")
    print("="*80)
    
    # Load configuration
    config = load_azure_config()
    
    print(f"\nConfiguration:")
    print(f"  Subscription ID: {config['subscription_id']}")
    print(f"  Resource Group: {config['resource_group']}")
    print(f"  Workspace: {config['workspace_name']}")
    print(f"  Compute Target: {compute_target}")
    
    # Authenticate
    print(f"\nAuthenticating...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        config['subscription_id'],
        config['resource_group'],
        config['workspace_name']
    )
    print(f"✓ Authenticated to Azure ML")
    
    # Resolve paths
    script_dir = Path(__file__).parent.parent
    src_path = script_dir / 'src'
    data_path = script_dir / 'data' / 'trades_synthetic.csv'
    env_file = script_dir / 'environment.yml'
    
    print(f"\nPaths:")
    print(f"  Source: {src_path}")
    print(f"  Data: {data_path}")
    print(f"  Environment: {env_file}")
    
    # Check paths exist
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    
    print(f"✓ All paths verified")
    
    # Create environment
    print(f"\nCreating environment...")
    env = Environment(
        name='trade-exception-env',
        conda_file=str(env_file),
        description='Environment for trade exception predictor'
    )
    
    # Create training script wrapper
    train_script = src_path / 'train_azure_wrapper.py'
    if not train_script.exists():
        print(f"  Creating train_azure_wrapper.py...")
        _create_train_wrapper(src_path)
    
    print(f"✓ Environment created")
    
    # Define training job
    print(f"\nDefining training job...")
    
    training_job = command(
        code=str(src_path),
        command=(
            f'python train_azure_wrapper.py '
            f'--n_trials {n_trials} '
            f'--test_size {test_size} '
            f'--random_state {random_state}'
        ),
        inputs={
            'training_data': Input(
                type='uri_file',
                path=str(data_path),
                description='Trade dataset (CSV)'
            )
        },
        outputs={
            'model': Output(
                type='uri_folder',
                description='Trained model artifacts'
            ),
            'metrics': Output(
                type='uri_folder',
                description='Training metrics and plots'
            )
        },
        environment=env,
        compute=compute_target,
        display_name=job_name,
        description='Train XGBoost model for trade exception prediction',
        tags={
            'project': 'trade_exception_predictor',
            'model': 'xgboost',
            'optimization': 'optuna',
            'explainability': 'shap'
        },
        properties={
            'dataset': 'trades_synthetic.csv',
            'n_trials': str(n_trials),
            'test_size': str(test_size)
        }
    )
    
    print(f"✓ Job definition created")
    
    # Submit job
    print(f"\nSubmitting job...")
    job = ml_client.create_or_update(training_job)
    
    print(f"\n" + "="*80)
    print(f"✓ JOB SUBMITTED SUCCESSFULLY")
    print(f"="*80)
    print(f"\nJob Details:")
    print(f"  Name: {job.name}")
    print(f"  ID: {job.id}")
    print(f"  Status: {job.status}")
    print(f"  Studio URL: {job.studio_url}")
    
    return job


def _create_train_wrapper(src_path: Path):
    """Create Azure ML training wrapper script."""
    
    wrapper_code = '''"""
train_azure_wrapper.py - Azure ML Training Wrapper

Wrapper script for running training on Azure ML compute.
Integrates all updated modules:
  - features_updated.py
  - train_updated.py
  - explain_updated.py
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import joblib
import mlflow

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from features_updated import FeatureEngineering
from train_updated import TradeExceptionPredictor
from explain_updated import ExplainabilityAnalyzer


def main():
    """Train model on Azure ML."""
    
    parser = argparse.ArgumentParser(description='Train trade exception model')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"\\n{'='*80}")
    print("AZURE ML TRAINING JOB")
    print(f"{'='*80}")
    
    # Load data (Azure provides via mounted inputs)
    print("\\nLoading data...")
    # In Azure, input data is mounted at /mnt/batch/inputs/training_data/
    data_path = Path('/mnt/batch/inputs/training_data/trades_synthetic.csv')
    
    if not data_path.exists():
        # Fallback for local testing
        data_path = Path('./data/trades_synthetic.csv')
    
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Feature engineering
    print("\\nFeature engineering...")
    fe = FeatureEngineering(df)
    X_train, X_test, y_train, y_test, feature_names = fe.engineer_features(
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Model training
    print("\\nModel training...")
    mlflow.set_experiment('trade_exception_predictor')
    
    with mlflow.start_run(run_name='azure_ml_training'):
        mlflow.set_tag('cloud_platform', 'Azure ML')
        mlflow.set_tag('optuna_trials', args.n_trials)
        
        predictor = TradeExceptionPredictor(n_trials=args.n_trials, random_state=args.random_state)
        predictor.train(X_train, y_train, X_test, y_test)
        
        # Log parameters
        mlflow.log_param('test_size', args.test_size)
        mlflow.log_param('random_state', args.random_state)
        
        # SHAP explanation
        print("\\nGenerating SHAP explanations...")
        analyzer = ExplainabilityAnalyzer(predictor.model, X_test, y_test)
        importance_df = analyzer.get_feature_importance(top_k=15)
        
        # Save artifacts
        print("\\nSaving artifacts...")
        output_dir = Path('/mnt/batch/outputs/model')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(predictor.model, output_dir / 'xgboost_model.pkl')
        joblib.dump(fe, output_dir / 'feature_engineer.pkl')
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        # Save to MLflow
        mlflow.xgboost.log_model(predictor.model, 'xgboost_model')
        mlflow.log_artifact(str(output_dir / 'feature_importance.csv'))
        
        print(f"✓ Artifacts saved to {output_dir}")
        print(f"\\n{'='*80}")
        print("✓ TRAINING COMPLETE")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
'''
    
    wrapper_path = src_path / 'train_azure_wrapper.py'
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_code)
    
    print(f"  ✓ Created {wrapper_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Submit training job to Azure ML')
    parser.add_argument(
        '--compute',
        default='cpu-cluster',
        help='Compute cluster name'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of Optuna trials'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set ratio'
    )
    parser.add_argument(
        '--job_name',
        default='trade_exception_training',
        help='Job name'
    )
    
    args = parser.parse_args()
    
    try:
        job = submit_training_job(
            compute_target=args.compute,
            n_trials=args.n_trials,
            test_size=args.test_size,
            job_name=args.job_name
        )
        print(f"\n✓ Job successfully submitted!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)