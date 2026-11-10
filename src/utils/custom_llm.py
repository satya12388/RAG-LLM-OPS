from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class OllamaLLM(DeepEvalBaseLLM):

    def __init__(self, model_name="gemma4:e2b"):
        self.model_name = model_name
        self.model = ChatOllama(
            model=model_name,
            temperature=1,
            reasoning=False,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        response = model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        model = self.load_model()
        response = await model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return self.model_name




class GroqLLM(DeepEvalBaseLLM):

    def __init__(self, model_name="openai/gpt-oss-20b"):

        self.model_name = model_name

        self.model = ChatGroq(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        response = model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        model = self.load_model()
        response = await model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return self.model_name
