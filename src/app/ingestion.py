import hashlib
import logging
from logging import INFO
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
from src.config.config import params

load_dotenv()

## Logger
logger = logging.getLogger(name="Ingestion logger")
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(fmt=formatter)
logger.addHandler(handler)
logger.setLevel(INFO)

## Initializing Models
embedder = OpenAIEmbeddings(
    model=params.Ingestion.embedding_model,
)

vector_store = Chroma(
    collection_name=params.Ingestion.embedding_store,
    embedding_function=embedder,
    persist_directory=Path("./embeddings").as_posix()
)

retreiver = vector_store.as_retriever(
    search_type=params.Ingestion.retriever_search_type,
    search_kwargs={"k": params.Ingestion.retriever_K}
)


def generate_deterministic_ids(chunks: list) -> list[str]:
    """
    Generate deterministic IDs combining source filename, chunk index, and content hash.
    Example: 'ceiling-fan-guide.mdx:chunk_0:4f5a3b2c1d'
    """
    doc_chunk_counter: dict[str, int] = {}
    ids: list[str] = []

    for chunk in chunks:
        source_name = Path(chunk.metadata.get("source", "doc")).name
        chunk_idx = doc_chunk_counter.get(source_name, 0)
        doc_chunk_counter[source_name] = chunk_idx + 1

        # Generate a short SHA-256 hash of the chunk's text content
        content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:10]

        chunk_id = f"{source_name}:chunk_{chunk_idx}:{content_hash}"
        ids.append(chunk_id)
        chunk.metadata["id"] = chunk_id

    return ids


def ingest_docs():
    ## Load Documents
    raw_doc_path = Path("./raw_docs")
    loader = DirectoryLoader(
        path=raw_doc_path,
        glob="*mdx",
        loader_cls=TextLoader)

    docs = loader.load()

    ## Chunking the docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=params.Ingestion.chunk_size,
        chunk_overlap=params.Ingestion.chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Total chunks created from files: {len(chunks)}")

    ## Generate deterministic IDs for each chunk
    ids = generate_deterministic_ids(chunks)

    ## Fetch existing IDs from Chroma vector store (Free local SQLite query)
    existing_ids = set(vector_store.get().get("ids", []))

    ## Filter only the new or modified chunks that don't exist in Chroma yet
    new_chunks = []
    new_ids = []
    for chunk, chunk_id in zip(chunks, ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    ## Send ONLY new chunks to OpenAI Embeddings
    if new_chunks:
        skipped_count = len(chunks) - len(new_chunks)
        logger.info(f"Embedding {len(new_chunks)} new/modified chunks (Skipped {skipped_count} unchanged chunks, saving API cost).")
        vector_store.add_documents(new_chunks, ids=new_ids)
        logger.info("Ingestion completed successfully.")
    else:
       logger.info("All chunks are already up-to-date in vector store. 0 embedding API calls made!")

if __name__ == "__main__":
    # Only runs when executing `python ingestion.py` directly, not on import
    ingest_docs()






