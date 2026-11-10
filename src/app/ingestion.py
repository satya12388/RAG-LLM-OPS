from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

## Initializing Models
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

vector_store = Chroma(
    "guide-embeddings",
    embedding_function=embedder,
    persist_directory=Path("./embeddings").as_posix()
)

retreiver = vector_store.as_retriever(
    search_kwargs = {"k":4}
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
    splitter = RecursiveCharacterTextSplitter(chunk_size = 2000,chunk_overlap = 200)
    chunks = splitter.split_documents(docs)
    print(f"The size of chunks is:{len(chunks)}")

    ## Storing them in vectorstore
    vector_store.add_documents(chunks)

if __name__ == "__main__":
    # Only runs when executing `python ingestion.py` directly, not on import
    ingest_docs()






