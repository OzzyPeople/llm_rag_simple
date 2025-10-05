from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.deps import rag

router = APIRouter()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=AskResponse)
async def ask_stock_question(payload: AskRequest):
    if not rag.qa:
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready yet.")
    try:
        result = rag.qa.invoke({"question": payload.question})
        # if your chain returns a string already:
        answer = result if isinstance(result, str) else str(result)
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

@router.post("/reload")
async def reload_index():
    """
    Rebuild vectorstore/retriever/llm/chain (e.g., after you update the PDF).
    """
    rag.warmup()
    return {"status": "ok", "message": "RAG pipeline reloaded"}
