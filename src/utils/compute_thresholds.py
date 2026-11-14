import numpy as np
import mlflow
import dagshub
import json
from pathlib import Path
from src.utils.ml_flow_utils import get_metrics_from_tags

ROOT_DIR = Path(__file__).parent.parent.parent
THRESHOLDS = ROOT_DIR/"thresholds.json"

## DAGS Setup for ML FLow
mlflow.set_tracking_uri("https://dagshub.com/satya12388/RAG-LLM-OPS.mlflow")
dagshub.init(repo_owner='satya12388', repo_name='RAG-LLM-OPS', mlflow=True)

experiment_id = mlflow.get_experiment_by_name("llmops-ragapp").experiment_id

def calculate_stds(metrics:list[dict[str,float]]):

    all_metric_values = []

    for metric in metrics:
        metric_names = list(metric.keys())
        metric_values = list(metric.values())

        all_metric_values.append(metric_values)

    matrix = np.array(all_metric_values)
    stds = np.std(matrix, axis=0, ddof=1).tolist()
    
    return {key: val for key, val in zip(metric_names, stds)}

def save_thresholds(path,drift_stds,regression_stds):
    if path.exists():
        with open(path,"r") as file:
            thresholds = json.load(file)
            drift_thresholds = thresholds.get("drift_thresholds")
            regression_thresholds = thresholds.get("regression_thresholds")

    new_thresholds = {}
    new_thresholds["drift_thresholds"] = drift_stds
    new_thresholds["regression_thresholds"] = regression_stds

    with open(path,"w") as file:
        json.dump(new_thresholds,file,indent=4)


drift_stds = calculate_stds(get_metrics_from_tags(experiment_id=experiment_id,tag='drift'))
regression_stds = calculate_stds(get_metrics_from_tags(experiment_id=experiment_id,tag='regression'))

save_thresholds(THRESHOLDS,drift_stds,regression_stds)

print(drift_stds,regression_stds)