from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking(
    pages: List[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Chunk the documents into smaller pieces for processing.
    Args:
        pages: List[Document] to be chunked
    Returns:
        List[Document] chunks
    """
    if pages is None:
        raise ValueError("chunking() received None. Did load_pdf return None?")
    if not isinstance(pages, list):
        raise TypeError(f"chunking() expected List[Document], got {type(pages).__name__}")
    if len(pages) == 0:
        # Nothing to split; return empty list (not None)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # better boundaries; optional
    )
    return splitter.split_documents(pages)

