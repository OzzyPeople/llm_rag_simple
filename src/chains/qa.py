from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel
from typing import Iterable

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant. Use ONLY the provided context to answer. "
     "If the answer is not in the context, say you don't know."),
    ("human",
     "Question:\n{question}\n\nContext:\n{context}\n\n"
     "Instructions: Answer concisely. Include brief citations like [source].")
])

def _format_docs(docs: Iterable) -> str:
    """Robust: accepts List[Document] or List[str]."""
    parts = []
    for d in docs:
        if isinstance(d, Document):
            meta = d.metadata or {}
            src = (
                meta.get("source")
                or meta.get("file_path")
                or meta.get("path")
                or meta.get("pdf_path")
                or "unknown_source"
            )
            page = meta.get("page")
            tag = f"[source: {src}{f'#p{page}' if page is not None else ''}]"
            parts.append(f"{d.page_content}\n{tag}")
        else:
            parts.append(str(d))
    return "\n\n---\n\n".join(parts)

def build_rag_qa_chain(retriever, llm):
    # 1) fan-out: retrieve docs and pass the question through
    gather = RunnableParallel(
        docs=RunnableLambda(lambda x: retriever.invoke(x["question"])),
        question=lambda x: x["question"],
    )

    # 2) shape to prompt inputs (IMPORTANT: read docs from the dict)
    format_step = RunnableLambda(
        lambda m: {"question": m["question"], "context": _format_docs(m["docs"])}
    )

    chain = (
        gather
        | format_step
        | QA_PROMPT   # ChatPromptTemplate expecting {question, context}
        | llm
        | (lambda msg: getattr(msg, "content", msg))  # be tolerant to LLM return shape
    )
    return chain
