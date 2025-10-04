from src.llm_poc.pipeline import build_stock_market_pipeline
from src.retrievers.stock_market import get_stock_market_retriever
from src.tools.stock_market_tool import retriever_tool
from src.llm_poc.llm import get_gemini_llm
from src.chains.qa import build_rag_qa_chain
from dotenv import load_dotenv
import json
from src.chains.qa import build_rag_qa_chain
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
API_KEY = os.getenv("MODEL_API_KEY")
CHROMA_DIR= os.getenv("CHROMA_DIR")
pdf_path = os.getenv("PDF_PATH")


#1
vectorstore = build_stock_market_pipeline(
        pdf_path=pdf_path,
        persist_directory=CHROMA_DIR,
        collection_name="stock_market"
    )

retriever = get_stock_market_retriever(vectorstore, k=5)


# 3) Get LLM
llm = get_gemini_llm(model="gemini-2.0-flash", temperature=0.2)

# 4) Build QA chain
qa = build_rag_qa_chain(retriever, llm)

# 5) Ask
question = "What is Meta's market capitalization?"
answer = qa.invoke({"question": question})
print(answer)


#poetry run python -m src.llm_poc

"""
if __name__ == "__main__":
    print("\nFinal result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
"""


