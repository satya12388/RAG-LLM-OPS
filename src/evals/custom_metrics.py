from deepeval.metrics.g_eval import Rubric
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams
from src.utils.custom_llm import GroqLLM

llm = GroqLLM()

rubric = [
    Rubric(score_range=(0,2), expected_outcome="Factually incorrect."),
    Rubric(score_range=(3,6), expected_outcome="Mostly correct."),
    Rubric(score_range=(7,9), expected_outcome="Correct but missing minor details."),
    Rubric(score_range=(10,10), expected_outcome="100% correct."),
]

correctness_metric = GEval(
    name="correctness metric",
    criteria="Evaluate the correctness of response by comparing actual and expected outpusts",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT,SingleTurnParams.EXPECTED_OUTPUT],
    model= llm,
    rubric=rubric,
    verbose_mode=True
)
