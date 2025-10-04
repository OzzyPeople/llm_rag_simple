from langchain_google_genai import ChatGoogleGenerativeAI
import os

def get_gemini_llm(model: str = "gemini-2.0-flash", api_key: str | None = None, temperature: float = 0.2):
    """
    Returns a ChatGoogleGenerativeAI LLM for RAG answering.
    Requires GOOGLE_API_KEY in .env
    """
    if api_key is None:
        api_key = os.getenv("MODEL_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not provided. "
                "Pass api_key=... or set GOOGLE_API_KEY in your environment."
            )

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )