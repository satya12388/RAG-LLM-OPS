from time import sleep
from rag import graph_agent
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from pathlib import Path

## paths
ROOT_DIR = Path(__file__).parent
GOLDEN_DIR = ROOT_DIR/"data_sets"/"goldens"/"rag_golden.json"
TEST_CASES_DIR = GOLDEN_DIR.parent.parent/"test_cases"
TEST_CASES_DIR.mkdir(parents=True,exist_ok=True)

## creating data set and adding goldens
data_set = EvaluationDataset()
data_set.add_goldens_from_json_file(
    file_path=GOLDEN_DIR
)

# calling LLM to create test cases

for i,golden in enumerate(data_set.goldens):
    final_state = graph_agent.invoke({"query":golden.input})
    sleep(5)
    print(f"/n======/nresponse: {final_state["response"]}")
    data_set.add_test_case(LLMTestCase(
        input = golden.input,
        actual_output=final_state["response"],
        expected_output=golden.expected_output,
        retrieval_context=[doc.page_content for doc in final_state["retreived_docs"]]
    ))
    print(f"testcase: {i} added succesfully")
    

## save dataset with test cases
data_set.save_as(
    file_type='json',
    directory=TEST_CASES_DIR,
    file_name="rag_llm_testcases",
    include_test_cases=True
)
