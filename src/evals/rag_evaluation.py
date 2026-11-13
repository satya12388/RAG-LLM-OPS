from deepeval.evaluate import evaluate,AsyncConfig,DisplayConfig,CacheConfig
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.dataset import EvaluationDataset
from langchain_openai import ChatOpenAI
from src.evals.custom_metrics import correctness_metric

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR_PATH = Path(__file__).parent.parent.parent
TEST_CASES_PATH = ROOT_DIR_PATH/"data_sets"/"test_cases"/"rag_llm_testcases.json"
REPORTS_PATH = ROOT_DIR_PATH/"reports"
JSON_REPORTS = REPORTS_PATH/"Json"
MD_REPORTS = REPORTS_PATH/"Markdown"

REPORTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)
JSON_REPORTS.mkdir(
    parents=True,
    exist_ok=True
)
MD_REPORTS.mkdir(
    parents=True,
    exist_ok=True
)

llm = "gpt-4.1-mini"

evaluation_data = EvaluationDataset()
evaluation_data.add_test_cases_from_json_file(
    file_path=TEST_CASES_PATH,
    input_key_name="input",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    retrieval_context_key_name="retrieval_context"
)

test_cases = evaluation_data.test_cases

metrics = [
    ContextualRelevancyMetric(model=llm),
    ContextualPrecisionMetric(model=llm),
    ContextualRecallMetric(model=llm),
    AnswerRelevancyMetric(model=llm),
    FaithfulnessMetric(model=llm),
    correctness_metric
]

evaluate(
    test_cases = test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=True,throttle_value=5,max_concurrent=3),
    cache_config=CacheConfig(write_cache=False, use_cache=False),
    display_config=DisplayConfig(
        results_folder=JSON_REPORTS.as_posix(),
        file_type='md',
        file_output_dir=MD_REPORTS.as_posix())
)


