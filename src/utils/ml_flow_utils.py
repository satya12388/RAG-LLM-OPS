from mlflow import MlflowClient
import json
import mlflow
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
JSON_FILE_PATH = ROOT_DIR / "historical_runs.json"



def log_run_info(run_id: str, run_name: str):
    if JSON_FILE_PATH.exists():
        with open(JSON_FILE_PATH, "r") as file:
            historical_runs = json.load(file)         
    else:
        historical_runs = []
        
    run_dict = {"run_id": run_id,
                "run_name": run_name}
    
    historical_runs.append(run_dict)
    
    with open(JSON_FILE_PATH, "w") as file:
        json.dump(historical_runs, file,indent=4)


def get_metrics_from_tags(experiment_id,tag):
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.phase = '{tag}'",
        output_format='list'
    )
    metrics = []
    for run in runs:
        metrics.append(run.data.metrics)
    return metrics

def get_metrics_by_runid(run_id):
    run = mlflow.get_run(run_id)
    metrics = run.data.metrics
    return metrics

def get_metrics_by_stage(experiment_id,staging_type):
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.stage = '{staging_type}'",
        output_format='list'
    )
    metrics = runs[0].data.metrics
    return metrics

def get_runs_by_stage(experiment_id,staging_type):
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.stage = '{staging_type}'",
        output_format='list'
    )
    return runs

def get_run_by_stage(experiment_id,staging_type):
    runs = get_runs_by_stage(experiment_id,staging_type)
    if len(runs)!=1:
        raise RuntimeError(
            f"more than one runs found for the {staging_type}, please correct them on MLFLOW UI"
        )
    return runs[0]

def set_stage(run_id,staging_type):
    MlflowClient().set_tag(run_id=run_id,key="stage",value=staging_type)

def del_stage(run_id):
    MlflowClient().delete_tag(run_id=run_id,key="stage")

def remove_stale_challengers(experiment_id,staging_type="challanger"):
    stale_runs = get_runs_by_stage(experiment_id,staging_type)
    for run in stale_runs:
        del_stage(run.info.run_id,staging_type)

    return len(stale_runs)

def promoter_challenger_logic(experiment_id):

    champion = get_run_by_stage(experiment_id,"champion")
    challenger = get_run_by_stage(experiment_id,"challenger")

    set_stage(champion.info.run_id,"archive")
    set_stage(challenger.info.run_id,"champion")

    return champion.info.run_id,challenger.info.run_id

def reject_challenger_logic(experiment_id):

    challenger = get_run_by_stage(experiment_id,"challenger")

    del_stage(challenger.info.run_id)

    return challenger.info.run_id




