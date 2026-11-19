import mlflow,dagshub
from src.utils.ml_flow_utils import reject_challenger_logic

## DAGS Setup for ML FLow
mlflow.set_tracking_uri("https://dagshub.com/satya12388/RAG-LLM-OPS.mlflow")
dagshub.init(repo_owner='satya12388', repo_name='RAG-LLM-OPS', mlflow=True)
experiment_id = mlflow.get_experiment_by_name("llmops-ragapp").experiment_id

challenger = reject_challenger_logic(experiment_id)

print(f"the champion tag was removed from {challenger}")
