#!/usr/bin/env python
"""
check_deployment_readiness.py - Pre-Deployment Verification

Checks all prerequisites for Azure ML deployment before attempting deploy.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=str(env_path))

print("\n" + "="*80)
print("DEPLOYMENT READINESS CHECK")
print("="*80)

checks = []


def check(name, condition, details=""):
    """Record a check result."""
    status = "[OK]" if condition else "[FAIL]"
    print(f"\n{status} {name}")
    if details:
        print(f"     {details}")
    checks.append((name, condition))
    return condition


# ============================================================================
# 1. Azure Configuration
# ============================================================================

print("\n" + "-"*80)
print("AZURE CONFIGURATION")
print("-"*80)

sub_id = os.getenv('subscription_id', '').strip().strip('"')
rg = os.getenv('resource_group_name', '').strip().strip('"')
ws = os.getenv('workspace_name', '').strip().strip('"')

check("Subscription ID configured", bool(sub_id),
      f"Value: {sub_id[:20]}..." if sub_id else "Missing")
check("Resource Group configured", bool(rg),
      f"Value: {rg}" if rg else "Missing (should be: azurexgb-rg)")
check("Workspace configured", bool(ws),
      f"Value: {ws}" if ws else "Missing (should be: trade-exception-ws)")

# ============================================================================
# 2. Project Files
# ============================================================================

print("\n" + "-"*80)
print("PROJECT FILES")
print("-"*80)

project_root = Path(__file__).parent

files_to_check = {
    'src/score.py': 'Scoring script (required)',
    'environment.yml': 'Environment spec (required)',
    'notebooks/02_modeling.ipynb': 'Training notebook (for reference)',
    'azure/register_model.py': 'Model registration script',
    'azure/deploy_endpoint.py': 'Deployment script',
    'CLAUDE.md': 'Project documentation',
}

for file_path, description in files_to_check.items():
    full_path = project_root / file_path
    exists = full_path.exists()
    check(f"{file_path}", exists,
          f"? {description}")

# ============================================================================
# 3. Trained Model Artifacts
# ============================================================================

print("\n" + "-"*80)
print("MODEL ARTIFACTS")
print("-"*80)

outputs_dir = project_root / 'outputs'
model_pkl = outputs_dir / 'xgboost_model.pkl'
fe_pkl = outputs_dir / 'feature_engineer.pkl'

check("outputs/ directory exists", outputs_dir.exists())

if outputs_dir.exists():
    check("xgboost_model.pkl present", model_pkl.exists(),
          f"Path: {model_pkl}")
    check("feature_engineer.pkl present", fe_pkl.exists(),
          f"Path: {fe_pkl}")

    if model_pkl.exists():
        size_mb = model_pkl.stat().st_size / (1024*1024)
        check("Model file size > 1MB", size_mb > 1,
              f"Size: {size_mb:.1f}MB")

    if fe_pkl.exists():
        size_mb = fe_pkl.stat().st_size / (1024*1024)
        check("Feature engineer size > 100KB", size_mb > 0.1,
              f"Size: {size_mb:.2f}MB")
else:
    print("\n[WARNING] outputs/ directory not found")
    print("   Create it and copy trained artifacts:")
    print("   mkdir -p outputs")
    print("   cp xgboost_model.pkl outputs/")
    print("   cp feature_engineer.pkl outputs/")

# ============================================================================
# 4. Python Dependencies
# ============================================================================

print("\n" + "-"*80)
print("PYTHON DEPENDENCIES")
print("-"*80)

required_packages = {
    'azure.ai.ml': 'Azure ML SDK',
    'azure.identity': 'Azure Identity',
    'azure.core': 'Azure Core',
    'xgboost': 'XGBoost',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'joblib': 'Joblib',
    'shap': 'SHAP',
    'scikit-learn': 'Scikit-learn',
}

for package, name in required_packages.items():
    try:
        __import__(package)
        check(name, True, f"Module: {package}")
    except ImportError:
        check(name, False, f"Missing: {package}")

# ============================================================================
# 5. Azure Connectivity
# ============================================================================

print("\n" + "-"*80)
print("AZURE CONNECTIVITY")
print("-"*80)

try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml import MLClient

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, sub_id, rg, ws)

    # Try to list workspaces
    try:
        ws_info = ml_client.workspaces.get(ws)
        check("Azure ML workspace accessible", True,
              f"Workspace: {ws_info.name}")
    except Exception as e:
        check("Azure ML workspace accessible", False,
              f"Error: {str(e)[:100]}")

except Exception as e:
    check("Azure SDK imports", False,
          f"Error: {str(e)[:100]}")

# ============================================================================
# 6. Scoring Script Validation
# ============================================================================

print("\n" + "-"*80)
print("SCORING SCRIPT VALIDATION")
print("-"*80)

score_script = project_root / 'src' / 'score.py'
if score_script.exists():
    with open(score_script, 'r') as f:
        content = f.read()

    check("Has init() function", 'def init():' in content)
    check("Has run() function", 'def run(' in content)
    check("Loads model with joblib", 'joblib.load' in content)
    check("Uses SHAP explainer", 'shap' in content or 'TreeExplainer' in content)

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

passed = sum(1 for _, result in checks if result)
total = len(checks)

print(f"\nChecks passed: {passed}/{total}")

if passed == total:
    print("\n[SUCCESS] All checks passed! Ready to deploy.")
    print("\nNext steps:")
    print("1. python azure/register_model.py")
    print("2. python azure/deploy_endpoint.py --action deploy")
    sys.exit(0)
else:
    print(f"\n[WARNING] Some checks failed. Please fix issues above before deploying.")
    print("\nFailing checks:")
    for name, result in checks:
        if not result:
            print(f"  [FAIL] {name}")
    sys.exit(1)
