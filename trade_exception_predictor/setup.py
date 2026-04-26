from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-sub-id",
    resource_group_name="your-rg",
    workspace_name="your-workspace")