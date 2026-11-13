from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig,EvolutionConfig,Evolution,ContextConstructionConfig
from pathlib import Path
from dotenv import load_dotenv
from src.config.config import params

load_dotenv()

## Paths
ROOT_PATH = Path(__file__).parent.parent.parent
DOCS_PATH  = ROOT_PATH/"raw_docs"
DATA_SET_PATH = ROOT_PATH/"data_sets"
DATA_SET_PATH.mkdir(parents = True,exist_ok = True)
GOLDEN_DATA_PATH = DATA_SET_PATH/"goldens"
GOLDEN_DATA_PATH.mkdir(parents = True,exist_ok = True)


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
    synthetic_input_quality_threshold=params.Golden.filtration_config.filtration_threshold,
    max_quality_retries=params.Golden.filtration_config.max_filtration_retries,
    critic_model=params.Golden.filtration_config.filtration_critic_model
)

evolutions = {
    getattr(Evolution, k.split(".")[-1].upper()): v
    for k, v in params.Golden.evolution_config.evolutions.items()
}

evolution_config = EvolutionConfig(
    num_evolutions=params.Golden.evolution_config.num_evolutions,
    evolutions=evolutions,
)

context_config = ContextConstructionConfig(
    max_contexts_per_document=params.Golden.context_config.max_context_per_doc,
    context_quality_threshold=params.Golden.context_config.context_threshold,
    critic_model=params.Golden.context_config.context_critic_model
)

synthesizer = Synthesizer(
    model=params.Golden.synthesizer_model,
    max_concurrent=params.Golden.max_concurrent,
    filtration_config=filtration_config,
    evolution_config=evolution_config
)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=get_doc_path(path=DOCS_PATH),
    include_expected_output=True,
    max_goldens_per_context=params.Golden.max_goldens_per_doc,
    context_construction_config=context_config
)

## Save goldens

synthesizer.save_as(
    file_type="json",
    directory=GOLDEN_DATA_PATH,
    file_name=params.Golden.golden_file_name
)
