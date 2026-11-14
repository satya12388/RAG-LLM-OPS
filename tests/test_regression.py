import json
import pytest
from pathlib import Path
from src.utils.ml_flow_utils import get_metrics_by_runid,get_metrics_by_stage
import mlflow
import dagshub

## PAths
ROOT_DIR = Path(__file__).parent.parent
THRESHOLDS = ROOT_DIR/"thresholds.json"
HISTORICAL_RUNS = ROOT_DIR/"historical_runs.json"

MULTIPLIER = 2
TARGET = "staging"
MIN_METRICS_TO_PASS = 3

## DAGS Setup for ML FLow
mlflow.set_tracking_uri("https://dagshub.com/satya12388/RAG-LLM-OPS.mlflow")
dagshub.init(repo_owner='satya12388', repo_name='RAG-LLM-OPS', mlflow=True)

experiment_id = mlflow.get_experiment_by_name("llmops-ragapp").experiment_id

def get_latest_runid(path:Path):
    if path.exists():
        with open(path,'r') as file:
            historical_runs = json.load(file)
        latest_run_id = historical_runs[-1]["run_id"]
    return latest_run_id

def get_threshold_values(path,type):
    if path.exists():
        with open(path,'r') as file:
            thresholds = json.load(file)

    if type == "drift_thresholds":
        return thresholds["drift_thresholds"]
    elif type == "regression_thresholds":
        return thresholds["regression_thresholds"]
    else:
        raise ValueError(f"only drift and regression thresholds available")

run_id = get_latest_runid(HISTORICAL_RUNS)
latest_metrics = get_metrics_by_runid(run_id)
staging_metrics = get_metrics_by_stage(experiment_id,TARGET)
drift_thresholds = get_threshold_values(THRESHOLDS,"drift_thresholds")
regression_thresholds = get_threshold_values(THRESHOLDS,"regression_thresholds")
metric_names = latest_metrics.keys()
staging_names = staging_metrics.keys()


def test_similar_metric_names():
    assert metric_names == staging_names, "comparison metrics different, use same metrics for comparison only"
    if not metric_names == staging_names:
        pytest.exit(reason="Comparison metrics are different")

@pytest.mark.parametrize(argnames="metric",argvalues=metric_names)
def test_regression_on_metric(metric):

    base_value = staging_metrics[metric]
    actual_value = latest_metrics[metric]
    threshold = MULTIPLIER *(drift_thresholds[metric]+regression_thresholds[metric])
    assert (actual_value > base_value-threshold), f'test case failed on metric: {metric}, actual:{actual_value}, base:{base_value}'


def test_promotion_on_majority_metrics():

    better_count = 0
    breached_metrics = []
    verdicts = []

    for metric in metric_names:
        base_value = staging_metrics[metric]
        actual_value = latest_metrics[metric]
        threshold = MULTIPLIER *(drift_thresholds[metric])

        delta = actual_value - base_value

        is_better_or_equal = delta >= 0
        is_within_tolerance = delta >= -threshold

        if is_better_or_equal:
            better_count+=1

        if not is_within_tolerance:
            breached_metrics.append(metric)

        status = "BETTER" if is_better_or_equal else "NOISE" if is_within_tolerance else "FAIL"
        verdicts.append(
            f"  [{status}] {metric!r}: latest={actual_value:.4f} champion={base_value:.4f} "
            f"delta={delta:+.4f} tolerance=+/-{threshold:.4f}"
        )

    detail = "\n".join(verdicts)

    assert not breached_metrics, (
        f"Promotion gate failed: metric(s) {breached_metrics} regressed beyond their noise "
        f"tolerance.\nPer-metric detail:\n{detail}"
    )
    assert better_count >= MIN_METRICS_TO_PASS, (
        f"Promotion gate failed: {better_count} metrics better-than-or-equal-to "
        f"champion (need >= {MIN_METRICS_TO_PASS}); remaining metrics must still fall "
        f"within noise tolerance.\nPer-metric detail:\n{detail}"
    )




