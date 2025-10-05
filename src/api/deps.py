from typing import Optional
from langchain.schema.runnable import Runnable
from langchain_core.retrievers import BaseRetriever


from src.llm_poc.pipeline import build_stock_market_pipeline
from src.retrievers.stock_market import get_stock_market_retriever
from src.llm_poc.llm import get_gemini_llm
from src.chains.qa import build_rag_qa_chain

from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

CHROMA_DIR = "chroma_db"
PDF_PATH = "data/Stock_Market_Performance_2024.pdf"

class RAGContainer:
    def __init__(self):
        self.vectorstore = None
        self.retriever: Optional[BaseRetriever] = None
        self.llm = None
        self.qa: Optional[Runnable] = None

    def warmup(self):
        # 1) index / load
        self.vectorstore = build_stock_market_pipeline(
            pdf_path=PDF_PATH,
            persist_directory=CHROMA_DIR,
            collection_name="stock_market",
        )
        # 2) retriever
        self.retriever = get_stock_market_retriever(self.vectorstore, k=5)
        # 3) llm
        self.llm = get_gemini_llm(model="gemini-2.0-flash", temperature=0.2)
        # 4) chain
        self.qa = build_rag_qa_chain(self.retriever, self.llm)

rag = RAGContainer()