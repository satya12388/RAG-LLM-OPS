from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from guardrails import AsyncGuard
from guardrails.validators import register_validator,Validator,ValidationResult,PassResult,FailResult
from guardrails_ai.prompt_injection_detector import PromptInjectionDetector
from guardrails_ai.relevancy_evaluator import RelevancyEvaluator
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import asyncio

load_dotenv()
guard_llm = "gpt-4o-mini"

## Custom models to use in guardrails
class TopicResult(BaseModel):
    score: float = Field(
        description="Relevance score from 0.0 (off-topic) to 1.0 (strongly on-topic)"
    )
    topic: str = Field(
        description="The exact matched topic from the valid list, or 'None' if off-topic"
    )

## custom guard rails on input
@register_validator(name="on_topic",data_type="string")
class OnTopic(Validator):
    def __init__(
        self,
        validate_topic:list,
        threshold = 0.7,
        on_fail:str|None = None,
    ):
        super().__init__(
            on_fail = on_fail,
            validate_topic =validate_topic,
            threshold = threshold
        )

        self.validate_topic = validate_topic
        self.threshold = threshold

        self.llm = ChatOpenAI(model=guard_llm,temperature=0).with_structured_output(TopicResult)
        self.prompt = PromptTemplate.from_template(
            """You are a topic classification validator.

            Evaluate whether the user query relates to any of the valid topics:
            - Valid Topics: {validate_topic}
            - User Query: "{query}"

            Instructions:
            1. Assign a relevance score `score` between 0.0 and 1.0:
            - 0.8 to 1.0: The query directly matches one of the valid topics.
            - 0.0 to 0.4: The query is off-topic, unrelated, or conversational chit-chat.
            2. Set `topic` to the exact matching topic from the list, or 'None' if off-topic."""
        )
        self.chain = self.prompt | self.llm

    def _validate(self,value:str,metadata:dict ={}):

        response = self.chain.invoke({
            "query":value,
            "validate_topic":','.join(self.validate_topic)
        })
        if response.score>=self.threshold and response.topic in self.validate_topic:
            return PassResult()
        else:
            return FailResult(
                error_message=(
                    f"Query '{value}' is off-topic. "
                    f"Best match: '{response.topic}' with score {response.score:.2f} "
                    f"(threshold: {self.threshold})."
                )
            )

topics = ["electrical and electrical terminologies","home appliances","buying guides related to electrical equipments and home appliances"]
input_guard = AsyncGuard().use(
    OnTopic(threshold=0.6,validate_topic=topics,on_fail="refrain"),
    PromptInjectionDetector(
        llm_callable=guard_llm,
        on_fail="exception"
    )
)

async def validate_input(query:str):
    return await input_guard.validate(query)

## Guard rails on retriever
retreiver_guard = AsyncGuard().use(
    PromptInjectionDetector(
        llm_callable=guard_llm,
        on_fail="exception"
    )
)

async def validate_retreiver(context:str):
    return await retreiver_guard.validate(context)

## Guard rails on output
output_guard = AsyncGuard().use(
    RelevancyEvaluator(
        llm_callable=guard_llm,
        on_fail="refrain"
    )
)

async def validate_response(query:str,response:str):
    metadata = {
        "original_prompt": query,
    }
    return await output_guard.validate(response,metadata = metadata)



