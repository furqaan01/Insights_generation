# context_retriever.py
import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.documents import Document # Still needed for loading metadata

# --- Configuration ---
# Adapt these paths if your vector store is saved elsewhere
VECTOR_STORE_DIR = "vector_data"
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss.index")
METADATA_FILE = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
EMBEDDING_MODEL = "models/text-embedding-004" # Use the same model as in embedder_gemini.py
EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY" # Use 'RETRIEVAL_QUERY' for searching
TOP_K = 3 # Number of relevant chunks to retrieve

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env file for context retriever")
try:
    genai.configure(api_key=api_key)
except Exception as e:
    raise RuntimeError(f"Failed to configure Gemini API: {e}")

class ContextRetriever:
    """
    Loads a vector store and retrieves relevant document context for queries.
    """
    def __init__(self, index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
        self.index: Optional[faiss.Index] = None
        self.documents: Optional[List[Document]] = None
        self._load_vector_store(index_path, metadata_path)

    def _load_vector_store(self, index_path: str, metadata_path: str):
        """Loads the FAISS index and metadata."""
        print("[ContextRetriever] Initializing...")
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"[ContextRetriever] Warning: Index file ({index_path}) or metadata file ({metadata_path}) not found. Retrieval will be disabled.")
            return # Allow retriever to exist but be non-functional if files are missing

        try:
            print(f"[ContextRetriever] Loading FAISS index from {index_path}...")
            self.index = faiss.read_index(index_path)
            print(f"[ContextRetriever] FAISS index loaded. Size: {self.index.ntotal}, Dim: {self.index.d}")

            print(f"[ContextRetriever] Loading metadata from {metadata_path}...")
            with open(metadata_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"[ContextRetriever] Metadata loaded. Documents: {len(self.documents)}")

            if self.index.ntotal != len(self.documents):
                print(f"[ContextRetriever] Warning: Index size ({self.index.ntotal}) != metadata size ({len(self.documents)}).")
            print("[ContextRetriever] Ready.")

        except Exception as e:
            print(f"[ContextRetriever] Error loading vector store: {e}")
            self.index = None
            self.documents = None # Ensure state reflects failure

    def _embed_query(self, query_text: str) -> Optional[List[float]]:
        """Embeds a single query string using the Gemini API."""
        if not query_text:
            return None
        try:
            # Embed a single query text
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=[query_text], # API expects a list, even for one item
                task_type=EMBEDDING_TASK_TYPE # Important for search queries
            )
            if "embedding" in result and result["embedding"]:
                return result["embedding"][0] # Return the first (only) embedding
            else:
                print(f"[ContextRetriever] Warning: Failed to get embedding for query. API Result: {result}")
                return None
        except Exception as e:
            print(f"[ContextRetriever] Error during query embedding: {e}")
            return None

    def _search_vector_store(self, query_embedding: List[float], k: int) -> List[Tuple[Document, float]]:
        """Performs similarity search using the loaded FAISS index."""
        if self.index is None or self.documents is None or not query_embedding:
            return []
        if k <= 0: k = 1
        if k > self.index.ntotal: k = self.index.ntotal

        try:
            query_np = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_np, k)
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                if 0 <= idx < len(self.documents):
                    results.append((self.documents[idx], float(dist)))
                else:
                     print(f"[ContextRetriever] Warning: Retrieved index {idx} out of bounds.")
            return results
        except Exception as e:
            print(f"[ContextRetriever] Error during vector search: {e}")
            return []

    def retrieve_context(self, query: str, num_chunks: int = TOP_K) -> str:
        """
        Embeds a query, searches the vector store, and returns formatted context.
        Returns an empty string if retrieval is not possible or fails.
        """
        if self.index is None or self.documents is None:
             print("[ContextRetriever] Retrieval skipped: Vector store not loaded.")
             return "" # Return empty string if store isn't loaded

        print(f"[ContextRetriever] Retrieving context for query: '{query[:50]}...'")
        query_embedding = self._embed_query(query)
        if not query_embedding:
            print("[ContextRetriever] Failed to embed query. No context retrieved.")
            return ""

        search_results = self._search_vector_store(query_embedding, k=num_chunks)

        if not search_results:
            print("[ContextRetriever] No relevant document chunks found.")
            return ""

        # Format results for the prompt
        context_str = "--- Relevant Information from Documents ---\n"
        for doc, score in search_results:
            context_str += f"[Source Chunk (Score: {score:.4f})]:\n{doc.page_content}\n---\n"

        print(f"[ContextRetriever] Retrieved {len(search_results)} chunks.")
        return context_str.strip()

# Example usage (optional, for testing)
if __name__ == "__main__":
    # This assumes you have run main.py first to create the vector store in ./vector_data
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        retriever = ContextRetriever()
        test_query = "What were the sales figures?"
        retrieved_context = retriever.retrieve_context(test_query)

        if retrieved_context:
            print("\n--- Retrieved Context ---")
            print(retrieved_context)
        else:
            print("\nNo context retrieved for the test query.")
    else:
        print("\nSkipping ContextRetriever test: Vector store files not found.")
        print(f"Please run main.py to generate '{INDEX_FILE}' and '{METADATA_FILE}'.")