from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph,START,END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List,TypedDict
from langfuse import get_client
from src.app.ingestion import retreiver
from dotenv import load_dotenv

load_dotenv()
langfuse = get_client()

system_prompt = langfuse.get_prompt(
    name="rag_system_prompt",
    label="staging"
)


llm = ChatOpenAI(
    model_name = "gpt-4o-mini",
    temperature=1,
)

class RagState(TypedDict):
    retreived_docs:List[Document]
    context:str
    prompt:str
    query:str
    response:str

def f_retreiver(state:RagState) -> dict:
    query = state["query"]
    retreived_docs = retreiver.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retreived_docs])
    return {"retreived_docs":retreived_docs,"context":context}

def f_augment(state:RagState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt.prompt),
        ("human","query:{query} and context:{context}")]
    )

    return {"prompt":prompt}

def f_generate(state:RagState):

    prompt = state["prompt"]
    query = state["query"]
    context = state["context"]

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"query":query,"context":context})

    return {"response":response}

graph = StateGraph(RagState)

graph.add_node("n_retreiver",f_retreiver)
graph.add_node("n_augment",f_augment)
graph.add_node("n_generate",f_generate)

graph.add_edge(START,"n_retreiver")
graph.add_edge("n_retreiver","n_augment")
graph.add_edge("n_augment","n_generate")
graph.add_edge("n_generate",END)

graph_agent = graph.compile()

# res = graph_agent.invoke({"query":"Does MCB protects humans from electric shock?"})

# print(f"Context retrieved: {res["context"]}")
# print(f"/n======/nresponse: {res["response"]}")