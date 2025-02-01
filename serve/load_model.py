import mlflow
import os

def load_latest_model():

    ip = os.getenv("mlflow_track_ip")
    ip = "https://" + ip

    mlflow.set_tracking_uri(ip) 
    # Define Model Name (same as in MLflow Registry)
    experiment_name = "AUTO_PIPELINE"
    model_name = experiment_name + "_MODEL"

    # Get the latest production model URI
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise ValueError(f"No versions of model '{model_name}' found in Production!")

    production_model_uri = f"models:/{model_name}/Production"
    print(f"Loading model from: {production_model_uri}")

    # Load the model
    model = mlflow.pyfunc.load_model(production_model_uri)
    return model
