from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
from src.config.config import params

load_dotenv()

## Initializing Models
embedder = OpenAIEmbeddings(
    model=params.Ingestion.embedding_model,
)

vector_store = Chroma(
    params.Ingestion.embedding_store,
    embedding_function=embedder,
    persist_directory=Path("./embeddings").as_posix()
)

retreiver = vector_store.as_retriever(
    search_type=params.Ingestion.retriever_search_type,
    search_kwargs={"k": params.Ingestion.retriever_K}
)


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
    print(f"The size of chunks is:{len(chunks)}")

    ## Storing them in vectorstore
    vector_store.add_documents(chunks)

if __name__ == "__main__":
    # Only runs when executing `python ingestion.py` directly, not on import
    ingest_docs()






