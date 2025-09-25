from src.clients.gemini_client import GeminiClient
from src.prompts.system_prompt import PROMPT_ANALYST
from src.prompts.prompt_task import *
from src.schemas.forecast import Forecast

import json
from src.embeddings.google_embed import get_google_embeddings
from src.clients.chroma_client import get_chroma_store
from src.utils.text_split import chunking
from src.utils.loader import load_pdf


def build_stock_market_pipeline(
    pdf_path: str,
    persist_directory: str ,
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




def forecast_simple(api_key: str, text: str) -> dict | str:
    # Initialize client
    client = GeminiClient(
        api_key=api_key,
        system_prompt=PROMPT_ANALYST
    )
    prompt = forcast_prompt(text)

    output = client.generate(
            prompt,
            response_schema=Forecast,
            response_mime_type="application/json",
            temperature=0.7, top_p=0.9, max_output_tokens=200,
        )
    # Try to parse clean JSON
    try:
        obj = json.loads(output) if isinstance(output, str) else output
        print("=== FORECAST RESULT ===")
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return obj
    except Exception:
        print("=== FORECAST RESULT (RAW) ===")
        print(output)
        return output