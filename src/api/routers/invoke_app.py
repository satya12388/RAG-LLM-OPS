import time
from fastapi import APIRouter,HTTPException,status
from fastapi.responses import StreamingResponse
from src.api.schemas import AppRequest
from src.app.rag import graph_agent

router = APIRouter(prefix="/chat")

def stream_response(text):

    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


@router.post("/")
def invoke_app(request:AppRequest):
    query = request.query

    try:
        result = graph_agent.invoke({"query":query})

    except Exception as e:
        print(e)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    response = result["response"]

    return  StreamingResponse(stream_response(response),media_type="text/plain")
