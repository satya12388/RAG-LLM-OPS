from fastapi import APIRouter
from src.api.schemas import HealthResponse

router = APIRouter(prefix="/health")

@router.get("/",response_model=HealthResponse)
def check_health():
    return {"message":"Rag App is up and running."}