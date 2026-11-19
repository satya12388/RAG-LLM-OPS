import asyncio
import logging
from logging import INFO
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph,START,END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List,TypedDict
from langfuse import get_client
from src.app.ingestion import retreiver
from src.config.config import params
from src.app.guard_rails import validate_input,validate_retreiver,validate_response,topics
from dotenv import load_dotenv

load_dotenv()
langfuse = get_client()

## Logger
logger = logging.getLogger(name="Rag App logger")
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(fmt=formatter)
logger.addHandler(handler)
logger.setLevel(INFO)
logger.propagate = False


system_prompt = langfuse.get_prompt(
    name="rag_system_prompt",
    label=params.Rag.prompt_label
)


llm = ChatOpenAI(
    model_name=params.Rag.response_model,
    temperature=params.Rag.temperature,
)

CRITICAL_FALLBACK_MESSAGE = (
    "Something went wrong while processing your request and it could not be "
    "completed safely. Please try again in a moment."
)

topics = topics
SOFT_FALLBACK_MESSAGE = (
    f'''I couldn't produce a fully reliable answer to that question. Could you 
    rephrase it or ask something more specific only related to topics:{topics}?'''
)

RESPONSE_OUT_OF_CONTEXT_MESSAGE = (
    f'''I couldn't produce a fully reliable answer to that question. As there is no context availble to me.'''
)


class RagState(TypedDict):
    retreived_docs:List[Document]
    context:str
    prompt:str
    query:str
    response:str
    guardrails_status:Literal['ok','exception','refrain']
    guardrails_stage:Literal['input','retreiver','output']
    guardrails_msg:str


def f_retreiver(state:RagState) -> dict:
    query = state["query"]
    retreived_docs = retreiver.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retreived_docs])
    logger.info("retreiver step completed")
    return {"retreived_docs":retreived_docs,"context":context}

def f_augment(state:RagState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt.prompt),
        ("human","query:{query} and context:{context}")]
    )
    logger.info("Prompt Augmentation step completed")
    return {"prompt":prompt}

def f_generate(state:RagState):

    prompt = state["prompt"]
    query = state["query"]
    context = state["context"]

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"query":query,"context":context})

    logger.info("Response Genereation step completed")
    return {"response":response}

## Guardrailnodes

def inputguardrailnode(state:RagState):
    query = state['query']

    try:
        outcome = asyncio.run(validate_input(query))
    except Exception as e:
        result = {
            "guardrails_status":"exception",
            "guardrails_stage":"input",
            "guardrail_msg":str(e)
        }
        logger.error(f"query failed at input guardrails:{e}")
        return result

    if not outcome.validation_passed:
        result = {
            "guardrails_status":"refrain",
            "guardrails_stage":"input",
            "guardrail_msg":outcome.error or "Input Validation Failed"
        }
        logger.error(f"input guardrails result:{result}")
        return result

    result = {
            "query":outcome.validated_output or query,
            "guardrails_status":"ok",
            "guardrails_stage":"input",
            "guardrail_msg":""
        }
    logger.info("Input Query Passed Validatin")
    return result

def retreivarguardrailnode(state:RagState):
    context = state['context']

    if not context.strip():
        result = {
            "guardrails_status": "refrain",
            "guardrails_stage": "retreiver",
            "guardrail_msg": "No relevant documents were retrieved.",
        }
        return result

    try:
        outcome = asyncio.run(validate_retreiver(context))
    except Exception as e:
        result = {
            "guardrails_status":"exception",
            "guardrails_stage":"retreiver",
            "guardrail_msg":str(e)
        }
        logger.error(f"failed at retreivar guardrails:{e}")
        return result

    if not outcome.validation_passed:
        result = {
            "guardrails_status":"refrain",
            "guardrails_stage":"retreiver",
            "guardrail_msg":outcome.error or "retreiver validation failed"
        }
        logger.error(f"failed at retreivar guardrails:{result}")
        return result

    result = {
            "context":outcome.validated_output or context,
            "guardrails_status":"ok",
            "guardrails_stage":"retreiver",
            "guardrail_msg":""
        }
    logger.info("retreivar guard rails step passed")
    return result

def outputguardrailnode(state:RagState):
    response = state['response']
    query = state['query']

    try:
        outcome = asyncio.run(validate_response(query=query,response=response))
    except Exception as e:
        result = {
            "guardrails_status":"exception",
            "guardrails_stage":"output",
            "guardrail_msg":str(e)
        }
        logger.error(f"failed at output guardrails:{e}")
        return result

    if not outcome.validation_passed:
        result = {
            "guardrails_status":"refrain",
            "guardrails_stage":"output",
            "guardrail_msg":outcome.error or "response validation failed"
        }
        logger.error(f"failed at retreivar guardrails:{result}")
        return result

    result = {
            "response":outcome.validated_output or response,
            "guardrails_status":"ok",
            "guardrails_stage":"output",
            "guardrail_msg":""
        }
    logger.info("Output guardrails passed.")
    return result

def critical_msg_node(state:RagState):
    return {
        "response": CRITICAL_FALLBACK_MESSAGE,
    }

def softfallback_msg_node(state:RagState):
    return {
        "response": SOFT_FALLBACK_MESSAGE,
    }

def contextfallback_msg_node(state:RagState):
    return {
        "response": RESPONSE_OUT_OF_CONTEXT_MESSAGE,
    }

def route_guardrails(state:RagState):
    return state['guardrails_status']
    


graph = StateGraph(RagState)

graph.add_node("n_retreiver",f_retreiver)
graph.add_node("n_augment",f_augment)
graph.add_node("n_generate",f_generate)
graph.add_node('n_inputguardrails',inputguardrailnode)
graph.add_node('n_retreivarguardrails',retreivarguardrailnode)
graph.add_node('n_outputguardrails',outputguardrailnode)
graph.add_node('n_critical_msg',critical_msg_node)
graph.add_node('n_softfallback_msg',softfallback_msg_node)
graph.add_node('n_context_fallback_msg',contextfallback_msg_node)


graph.add_edge(START,"n_inputguardrails")
graph.add_conditional_edges("n_inputguardrails",
route_guardrails,
{
    "ok":"n_retreiver",
    "exception":"n_critical_msg",
    "refrain":"n_softfallback_msg"
})

graph.add_edge("n_retreiver","n_retreivarguardrails")
graph.add_conditional_edges("n_retreivarguardrails",
route_guardrails,
{
    "ok":"n_augment",
    "exception":"n_critical_msg",
    "refrain":"n_softfallback_msg"
})

graph.add_edge("n_augment","n_generate")
graph.add_edge("n_generate","n_outputguardrails")

graph.add_conditional_edges("n_outputguardrails",
route_guardrails,
{
    "ok":END,
    "exception":"n_critical_msg",
    "refrain":"n_context_fallback_msg"
})

graph.add_edge("n_critical_msg",END)
graph.add_edge("n_softfallback_msg",END)
graph.add_edge("n_context_fallback_msg",END)


graph_agent = graph.compile()

# res = graph_agent.invoke({"query":"what are Air Conditioner Buying guides?"})
# print(f"\n======\nresponse: {res["response"]}")