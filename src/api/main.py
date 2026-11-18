from src.api.routers import health,invoke_app
from fastapi import FastAPI

app = FastAPI(title = "LLMOps-RagApp")

app.include_router(health.router)
app.include_router(invoke_app.router)
