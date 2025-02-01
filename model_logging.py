import mlflow
import os
from datetime import datetime
import json
import pytz

# Get MLflow Tracking Server IP from environment variable
ip = os.getenv("mlflow_track_ip")
if ip:
    ip = "https://" + ip
else:
    raise ValueError("Environment variable 'mlflow_track_ip' is not set")

# Set MLflow Tracking URI
mlflow.set_tracking_uri(ip)

EXPERIMENT_NAME = "AUTO_PIPELINE"
MODEL_NAME = EXPERIMENT_NAME + "_MODEL"  # Dynamic model name based on the experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Define paths
model_path = 'serve/rf_model.pkl'  # Path to your pre-trained model
best_params = 'best_params.json'
metrics_path = 'evaluation_results.json'

# Validate required files exist
for file in [model_path, best_params, metrics_path]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Required file not found: {file}")

# Get current time in Baku timezone
baku_tz = pytz.timezone('Asia/Baku')
current_time = datetime.now(baku_tz).strftime("%Y-%m-%d_%H:%M:%S")
run_name = f"{current_time}_run"

with mlflow.start_run(run_name=run_name) as run:
    # Log artifacts (model, best params, and dvc.lock)
    mlflow.log_artifact(model_path, artifact_path="models")
    mlflow.log_artifact(best_params, artifact_path="models")
    mlflow.log_artifact("dvc.lock", artifact_path="models")

    # Log evaluation metrics
    with open(metrics_path, 'r') as f:
        params = json.load(f)
        for key, value in params.items():
            if isinstance(value, (int, float)):  # Log only numeric values as metrics
                mlflow.log_metric(key, value)

    # Get Run ID
    run_id = run.info.run_id

    # Model registry URI (Artifacts path for this run)
    model_uri = f"runs:/{run_id}/models"

    # Register the model
    registered_model = mlflow.register_model(model_uri, MODEL_NAME)
    model_version = registered_model.version

    print(f"Model registered with name: {MODEL_NAME}, version: {model_version}")

    # Promote model to Production
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Production"
    )
    print(f"Model {MODEL_NAME} version {model_version} is now in Production!")
