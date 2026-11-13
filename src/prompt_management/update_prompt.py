from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

langfuse.update_prompt(
    name = 'rag_system_prompt',
    version=3,
    new_labels=["staging"]
)