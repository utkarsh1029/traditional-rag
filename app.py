from src.data_loader import load_all_documents
from src.search import RAGSearch
from src.vector_store import FaissVectorStore


if __name__ == "__main__":
    #docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "What are the key points in the budget speech?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)