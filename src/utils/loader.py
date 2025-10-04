import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF into a list of LangChain Document objects.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Document]: Extracted documents, one per page.

    Raises:
        FileNotFoundError: If the given file path does not exist.
        RuntimeError: If PDF loading fails.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)

    try:
        pages: List[Document] = loader.load()
        print(f"[INFO] PDF loaded successfully: {len(pages)} pages from '{pdf_path}'")
        return pages
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF '{pdf_path}': {e}")

