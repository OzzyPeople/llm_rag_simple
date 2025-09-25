import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdf (pdf_path: str):
    # Safety measure
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_loader = PyPDFLoader(pdf_path)

    # Checks if the PDF is there
    try:
        pages = pdf_loader.load()
        print(f"PDF has been loaded and has {len(pages)} pages")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise