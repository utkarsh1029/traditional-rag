import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "faiss_store"

load_dotenv(PROJECT_ROOT / ".env")

class RAGSearch:
    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
    ):
        persist_dir = Path(persist_dir)
        self.vectorstore = FaissVectorStore(str(persist_dir), embedding_model)

        # Load or build vectorstore
        faiss_path = persist_dir / "faiss.index"
        meta_path = persist_dir / "metadata.pkl"
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents(str(DATA_DIR))
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to your project .env file.")

        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke(prompt)
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are the key points in the budget speech?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
