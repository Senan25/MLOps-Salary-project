import mlflow
import os
from datetime import datetime

ip = os.getenv("mlflow_track_ip")

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your remote tracking server URI
EXPERIMENT_NAME = "AUTO_PIPELINE"
mlflow.set_experiment(EXPERIMENT_NAME)

# Define paths
model_path = 'rf_model.pkl'  # Path to your pre-trained model
best_params = 'best_params.json'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"{current_time}_run"

with mlflow.start_run(run_name=run_name) as run:
    # Log the model
    mlflow.log_artifact(model_path, artifact_path="models")
    mlflow.log_artifact(best_params, artifact_path="models")

    print(f"Run ID: {run.info.run_id}")
