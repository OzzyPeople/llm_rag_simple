from langchain_google_genai import ChatGoogleGenerativeAI

def get_gemini_llm(model: str = "gemini-2.0-flash", temperature: float = 0.2):
    """
    Returns a ChatGoogleGenerativeAI LLM for RAG answering.
    Requires GOOGLE_API_KEY in .env
    """
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
    )