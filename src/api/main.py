from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routes import stock
from src.api.deps import rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    # build once on startup
    rag.warmup()
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="POC RAG API", version="1.0.0", lifespan=lifespan)
    app.include_router(stock.router, prefix="/stock", tags=["Stock QA"])

    @app.get("/")
    async def root():
        return {"message": "RAG service is up"}
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8080, reload=True)
