import json
from src.embeddings.google_embed import get_google_embeddings
from src.clients.chroma_client import get_chroma_store
from src.utils.text_split import chunking
from src.utils.loader import load_pdf


def build_stock_market_pipeline(
    pdf_path: str,
    persist_directory: str,
    collection_name: str = "stock_market"
):
    """
    Builds the stock market RAG pipeline: embeddings -> vectorstore
    Returns:
        vectorstore
    """
    # Step 1: Create embeddings (compatible with Gemini)
    embeddings = get_google_embeddings()
    pages_split = chunking(load_pdf(pdf_path))

    # Step 2: Store documents in Chroma vectorstore
    vectorstore = get_chroma_store(
        pages_split,
        embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    return vectorstore