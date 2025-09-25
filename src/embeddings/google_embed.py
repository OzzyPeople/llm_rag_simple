from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_google_embeddings(model: str = "models/embedding-001"):
    """
    Returns a GoogleGenerativeAIEmbeddings instance compatible with Gemini models.

    Args:
        model (str): The embedding model to use. Default = "models/embedding-001".
                     Other option: "models/text-embedding-004".
    """
    return GoogleGenerativeAIEmbeddings(model=model)