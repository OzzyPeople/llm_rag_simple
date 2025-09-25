
def get_stock_market_retriever(vectorstore, k: int = 5):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )