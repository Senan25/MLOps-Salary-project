import mlflow
import os
from datetime import datetime
import json

ip = os.getenv("mlflow_track_ip")
ip = "https://" + ip
print(ip)

# MLflow configuration
mlflow.set_tracking_uri(ip)  # Replace with your remote tracking server URI
EXPERIMENT_NAME = "AUTO_PIPELINE"
mlflow.set_experiment(EXPERIMENT_NAME)

# Define paths
model_path = 'serve/rf_model.pkl'  # Path to your pre-trained model
best_params = 'best_params.json'
metrics_path = 'evaluation_results.json'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(best_params):
    raise FileNotFoundError(f"Best parameters file not found at {best_params}")

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"{current_time}_run"

with mlflow.start_run(run_name=run_name) as run:
    # Log the model
    mlflow.log_artifact(model_path, artifact_path="models")
    mlflow.log_artifact(best_params, artifact_path="models")
    mlflow.log_artifact("dvc.lock", artifact_path="models")

    with open(metrics_path, 'r') as f:
        params = json.load(f)
        for key, value in params.items():
            if isinstance(value, (int, float)):  # Log only numeric values as metrics
                mlflow.log_metric(key, value)   

    print(f"Run ID: {run.info.run_id}")
