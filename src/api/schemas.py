from pydantic import BaseModel

class HealthResponse(BaseModel):
    message:str

class AppRequest(BaseModel):
    query:str