import mlflow
import os

def save_latest_model():

    ip = os.getenv("mlflow_track_ip")
    ip = "https://" + ip

    mlflow.set_tracking_uri(ip) 
    experiment_name = "AUTO_PIPELINE"
    # Experiment ID'yi al
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' bulunamadı.")
    
    experiment_id = experiment.experiment_id

    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        raise ValueError(f"Experiment '{experiment_name}' için herhangi bir run bulunamadı.")
    
    latest_run_id = runs.iloc[0]['run_id']
    print(f"En son Run ID: {latest_run_id}")

    # Model URI oluştur
    model_uri = f"runs:/{latest_run_id}/model"
    
    # Modeli yükle
    model = mlflow.pyfunc.load_model(model_uri)

    print(type(model))

    return model
