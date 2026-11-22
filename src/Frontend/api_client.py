import os
from typing import Iterator, Optional, Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")


class ApiClient:
    """Client to interact with the FastAPI RAG backend."""

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Check if the backend API is up and running.
        Returns a dict with 'status' ('ok', 'error', 'unreachable') and details.
        """
        url = f"{self.base_url}/health/"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return {
                    "status": "ok",
                    "data": response.json(),
                    "message": "Connected to backend API"
                }
            return {
                "status": "error",
                "message": f"Backend returned status {response.status_code}: {response.text}"
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "unreachable",
                "message": f"Unable to connect to backend at {self.base_url}. Please ensure the API is running."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }

    def stream_chat(self, query: str, session_id: Optional[str] = None) -> Iterator[str]:
        """
        Send query to the /chat/ endpoint and yield streamed text response chunks.
        """
        url = f"{self.base_url}/chat/"
        payload: Dict[str, Any] = {"query": query}
        if session_id:
            payload["session_id"] = session_id

        try:
            with requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.timeout
            ) as response:
                if response.status_code != 200:
                    yield f"Error from backend API ({response.status_code}): {response.text}"
                    return

                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk

        except requests.exceptions.ConnectionError:
            yield (
                "⚠️ Could not connect to the backend API at "
                f"`{self.base_url}`. Please make sure the backend is running."
            )
        except requests.exceptions.Timeout:
            yield "⚠️ Request to backend API timed out. Please try again."
        except Exception as e:
            yield f"⚠️ An unexpected error occurred: {str(e)}"

    def chat(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Send query to the /chat/ endpoint and return the full accumulated response.
        """
        return "".join(self.stream_chat(query=query, session_id=session_id))
