from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunking(pages):
    """
    Chunk the documents into smaller pieces for processing.

    Args:
        pages: List of documents to be chunked

    Returns:
        List of chunked documents
    """
    # Chunking Process
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    pages_split = text_splitter.split_documents(pages)
    return pages_split

