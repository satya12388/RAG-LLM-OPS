from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig,EvolutionConfig,Evolution,ContextConstructionConfig
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

## Paths
ROOT_PATH = Path(__file__).parent.parent.parent
DOCS_PATH  = ROOT_PATH/"raw_docs"
DATA_SET_PATH = ROOT_PATH/"data_sets"
DATA_SET_PATH.mkdir(parents = True,exist_ok = True)
GOLDEN_DATA_PATH = DATA_SET_PATH/"goldens"
GOLDEN_DATA_PATH.mkdir(parents = True,exist_ok = True)

llm = "gpt-5-nano"
critic_model = "gpt-4.1-mini"


def get_doc_path(path=DOCS_PATH):

    if not path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {path}"
        )

    document_paths = [
        doc.as_posix()
        for doc in path.rglob("*.mdx")
        if doc.is_file()
    ]

    print(f"Documents found: {len(document_paths)}")

    for doc in document_paths:
        print(f"  - {doc}")

    if not document_paths:
        raise ValueError(
            f"No .mdx files found inside: {path}"
        )

    return document_paths



filtration_config = FiltrationConfig(
    synthetic_input_quality_threshold=0.6,
    max_quality_retries=2,
    critic_model=critic_model
)

evolution_config = EvolutionConfig(
    num_evolutions=1,
    evolutions={
        Evolution.MULTICONTEXT:0.1,
        Evolution.CONSTRAINED:0.2,
        Evolution.CONCRETIZING:0.35,
        Evolution.REASONING:0.35,
    },
)

context_config = ContextConstructionConfig(
    max_contexts_per_document=2,
    context_quality_threshold=0.6,
    critic_model=critic_model
)

synthesizer = Synthesizer(
    model=llm,
    max_concurrent=1,
    filtration_config=filtration_config,
    evolution_config=evolution_config
)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=get_doc_path(path=DOCS_PATH),
    include_expected_output=True,
    max_goldens_per_context=1,
    context_construction_config=context_config
)

## Save goldens

synthesizer.save_as(
    file_type="json",
    directory=GOLDEN_DATA_PATH,
    file_name="rag_golden"
)
