import mlflow
import dagshub
from src.config.config import params
from src.data_generators.llm_test_case_generator import evaluation_dataset_generator
from src.evals.rag_evaluation import evaluate_app
from src.app.ingestion import ingest_docs
from src.utils.ml_flow_utils import log_run_info
from pathlib import Path
from langfuse import get_client
from dotenv import load_dotenv
import json
import logging
from logging import INFO


load_dotenv()
ROOT_DIR = Path(__file__).parent

## function to flatten params
def flatten_param(params):
    log_params = {}

    for key,value in params.items():
        if isinstance(value,dict):
            log_params.update(flatten_param(value))
        else:
            if key in log_params:
                raise ValueError(f"duplicate key found:{key}")
            log_params[key] = value
    return log_params

def get_latest_result_name(path:Path):
    file_names = []
    files = path.glob("*.json")
    for file in files:
        file_names.append(file.stem)
    recent_file = max(file_names)
    return recent_file

def get_latest_report_name(path:Path):
    file_names = []
    files = path.glob("*.md")
    for file in files:
        file_names.append(file.stem)
    recent_file = max(file_names)
    return recent_file
    
def get_aggregate_metrics(path:Path):

    metrics = {}
    with open(path,"r") as file:
        results = json.load(file)["metricsScores"]
    for result in results:
        metric_name = result["metric"].removesuffix(" [GEval]") if " [GEval]" in result["metric"] else result["metric"]
        scores = result["scores"]
        avg = round((sum(scores)/len(scores)),2)
        metrics[metric_name] = avg

    return metrics

def log_system_prompt(label = "staging"):
    langfuse = get_client()

    prompt = langfuse.get_prompt(
        name="rag_system_prompt",
        label=label
    )

    return prompt.prompt

if __name__ == "__main__":

    ## Logger
    logger = logging.getLogger(name="MLflow logger")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt=formatter)
    logger.addHandler(handler)
    logger.setLevel(INFO)
    

    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

    ## DAGS Setup for ML FLow
    mlflow.set_tracking_uri("https://dagshub.com/satya12388/RAG-LLM-OPS.mlflow")
    dagshub.init(repo_owner='satya12388', repo_name='RAG-LLM-OPS', mlflow=True)

    mlflow.set_experiment("llmops-ragapp")

    with mlflow.start_run() as run:

        ## log parameters

        all_params = flatten_param(params.model_dump())
        mlflow.log_params(all_params)
        logger.info("parameter logged")

        ## log prompt
        prompt = log_system_prompt(params.Rag.prompt_label)
        mlflow.log_text(prompt,artifact_file="sys_prompt.txt")
        logger.info("system prompt loggged")

        ## create evaluation dataset
        ingest_docs()
        evaluation_dataset_generator()
        logger.info("evaluation data set generated")
        evaluate_app()
        logger.info("evaluation completed")

        ## Paths to get results
        DATA_SET_PATH = ROOT_DIR/"data_sets"/"test_cases"
        RESULT_PATH = ROOT_DIR/"reports"/"Json"
        REPORT_PATH = RESULT_PATH.parent/"Markdown"
        recent_result_name = get_latest_result_name(RESULT_PATH)
        recent_report_name = get_latest_report_name(REPORT_PATH)

        result_file = RESULT_PATH/f"{recent_result_name}.json"
        report_file = REPORT_PATH/f"{recent_report_name}.md"
        test_case_file = DATA_SET_PATH/f"{params.LLM_TestCases.testcase_file_name}.json"

        mlflow.log_artifact(result_file,artifact_path="results")
        mlflow.log_artifact(report_file,artifact_path="reports")
        mlflow.log_artifact(test_case_file,artifact_path="data")
        logger.info("artifacts logged")

        ### log metrics
        metrics = get_aggregate_metrics(result_file)
        mlflow.log_metrics(metrics)
        logger.info("Metrics Logged")

    run_id = run.info.run_id
    run_name = run.info.run_name

    log_run_info(run_id, run_name)

    












