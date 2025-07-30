from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Azure ML workspace identifiers
subscription_id = 'ad017e07-0d6f-4cad-8669-16fd3fe4f33c'
resource_group = 'rg-dante.garciaperez-6665'
workspace = 'lucido'

# Get a handle to the workspace
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

file_model = Model(
    name="Qwen3-Embedding-06B",
    path="Qwen3-Embedding-06B",
    type=AssetTypes.CUSTOM_MODEL,
    version='1',
    description="Hugging Face Qwen3-Embedding-0.6B",
)

ml_client.models.create_or_update(file_model)