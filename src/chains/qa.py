from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant. Use ONLY the provided context to answer. "
     "If the answer is not in the context, say you don't know."),
    ("human",
     "Question:\n{question}\n\nContext:\n{context}\n\n"
     "Instructions: Answer concisely. Include brief citations like [source].")
])

def _format_docs(docs: List[Document]) -> str:
    chunks = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown_source"
        chunks.append(f"{d.page_content}\n[source: {src}]")
    return "\n\n---\n\n".join(chunks)

def build_rag_qa_chain(retriever, llm):
    """
    Returns a runnable chain: {"question": str} -> str
    """
    retriever_step = RunnableLambda(lambda x: retriever.invoke(x["question"]))
    format_step = RunnableLambda(lambda docs: {"context": _format_docs(docs)})

    # parallel: get docs + keep question
    gather = RunnableParallel(
        docs=retriever_step,
        question=lambda x: x["question"],
    )

    # map to prompt inputs
    to_prompt = (format_step | (lambda m: {"question": m["question"], "context": m["context"]}))

    chain = gather | to_prompt | QA_PROMPT | llm | (lambda msg: msg.content)
    return chain
