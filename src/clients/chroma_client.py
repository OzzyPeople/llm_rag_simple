import os
from langchain_chroma import Chroma

def get_chroma_store(pages_split, embeddings, persist_directory, collection_name):
    try:
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print("✅ Created ChromaDB vector store!")
        return vectorstore
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {str(e)}")
        raise
