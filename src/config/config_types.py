from typing import Literal, Dict
from pydantic import BaseModel, Field


class Ingestion(BaseModel):
    embedding_model: str
    embedding_store: str
    retriever_search_type: Literal["similarity", "mmr"]
    retriever_K: int
    chunk_size: int
    chunk_overlap: int

    model_config = {"extra": "forbid"}


class Rag(BaseModel):
    response_model: str
    temperature: float
    prompt_label:str

    model_config = {"extra": "forbid"}


class FiltrationConfig(BaseModel):
    filtration_threshold: float
    max_filtration_retries: int
    filtration_critic_model: str

    model_config = {"extra": "forbid"}


class EvolutionConfig(BaseModel):
    num_evolutions: int
    evolutions: Dict[str, float]

    model_config = {"extra": "forbid"}


class ContextConfig(BaseModel):
    context_threshold: float
    max_context_per_doc: int
    context_critic_model: str

    model_config = {"extra": "forbid"}


class Golden(BaseModel):
    synthesizer_model: str
    max_concurrent: int
    max_goldens_per_doc: int
    golden_file_name: str
    filtration_config: FiltrationConfig
    evolution_config: EvolutionConfig
    context_config: ContextConfig

    model_config = {"extra": "forbid"}


class LLM_TestCases(BaseModel):
    testcase_file_name: str

    model_config = {"extra": "forbid"}


class AsyncConfig(BaseModel):
    throttle_value: int
    max_concurrent_evaluations: int

    model_config = {"extra": "forbid"}


class Evaluation(BaseModel):
    judge_model: str
    async_config: AsyncConfig

    model_config = {"extra": "forbid"}


class ParamsConfig(BaseModel):
    Ingestion: Ingestion
    Rag: Rag
    Golden: Golden
    LLM_TestCases: LLM_TestCases
    Evaluation: Evaluation

    model_config = {"extra": "forbid"}
