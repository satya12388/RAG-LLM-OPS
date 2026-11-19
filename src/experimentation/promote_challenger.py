import mlflow,dagshub
from src.utils.ml_flow_utils import promoter_challenger_logic


## DAGS Setup for ML FLow
mlflow.set_tracking_uri("https://dagshub.com/satya12388/RAG-LLM-OPS.mlflow")
dagshub.init(repo_owner='satya12388', repo_name='RAG-LLM-OPS', mlflow=True)
experiment_id = mlflow.get_experiment_by_name("llmops-ragapp").experiment_id

old_champ,new_champ = promoter_challenger_logic(experiment_id)

print(f"The old champion stage for {old_champ} was set to archive and new champion tag was set to {new_champ}")
