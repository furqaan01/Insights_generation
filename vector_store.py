# vector_store.py
import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional
from langchain_core.documents import Document

# Configuration
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_metadata.pkl" # To store corresponding texts/docs

def create_vector_store(documents: List[Document], embeddings: List[List[float]], index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
    """
    Creates and saves a FAISS index and corresponding metadata.

    Args:
        documents: List of LangChain Document objects (used for metadata).
        embeddings: List of embedding vectors corresponding to the documents.
        index_path: Path to save the FAISS index file.
        metadata_path: Path to save the metadata (documents).
    """
    if not documents or not embeddings or len(documents) != len(embeddings):
        print("Error: Mismatch between documents and embeddings or empty inputs.")
        return

    dimension = len(embeddings[0])
    print(f"Creating FAISS index with dimension {dimension} for {len(documents)} documents.")

    # Use IndexFlatL2 for simple Euclidean distance search
    index = faiss.IndexFlatL2(dimension)

    # Convert embeddings to numpy array
    embeddings_np = np.array(embeddings).astype('float32')

    # Add vectors to the index
    index.add(embeddings_np)

    print(f"FAISS index created. Index size: {index.ntotal}")

    # Save the index
    try:
        faiss.write_index(index, index_path)
        print(f"FAISS index saved to {index_path}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
        return # Don't save metadata if index saving fails

    # Save the corresponding documents as metadata
    try:
        with open(metadata_path, 'wb') as f:
            pickle.dump(documents, f)
        print(f"Metadata (documents) saved to {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata: {e}")


def load_vector_store(index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE) -> Optional[Tuple[faiss.Index, List[Document]]]:
    """
    Loads a FAISS index and corresponding metadata.

    Args:
        index_path: Path to the FAISS index file.
        metadata_path: Path to the metadata (documents) file.

    Returns:
        A tuple containing the loaded FAISS index and the list of documents,
        or None if loading fails.
    """
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print(f"Error: Index file ({index_path}) or metadata file ({metadata_path}) not found.")
        return None

    try:
        print(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        print(f"FAISS index loaded. Index size: {index.ntotal}, Dimension: {index.d}")

        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            documents = pickle.load(f)
        print(f"Metadata loaded. Number of documents: {len(documents)}")

        # Basic validation
        if index.ntotal != len(documents):
             print(f"Warning: Mismatch between index size ({index.ntotal}) and number of documents ({len(documents)}).")
             # Decide how to handle mismatch - return None or proceed with caution
             # return None

        return index, documents

    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


def search_vector_store(query_embedding: List[float], index: faiss.Index, documents: List[Document], k: int = 5) -> List[Tuple[Document, float]]:
    """
    Performs a similarity search on the vector store.

    Args:
        query_embedding: The embedding vector for the query.
        index: The loaded FAISS index.
        documents: The list of documents corresponding to the index vectors.
        k: The number of nearest neighbors to retrieve.

    Returns:
        A list of tuples, each containing a retrieved Document and its similarity score (distance).
        Returns an empty list if search fails or inputs are invalid.
    """
    if not query_embedding or index is None or documents is None:
        print("Error: Invalid input for search (query_embedding, index, or documents missing).")
        return []
    if k > index.ntotal:
        print(f"Warning: Requested k={k} is larger than index size={index.ntotal}. Setting k={index.ntotal}.")
        k = index.ntotal
    if k <= 0:
         print("Warning: k must be greater than 0. Setting k=1.")
         k = 1


    try:
        # Convert query embedding to numpy array
        query_np = np.array([query_embedding]).astype('float32') # Needs to be 2D array

        # Perform the search
        distances, indices = index.search(query_np, k)

        # Retrieve results
        results = []
        for i in range(k):
            idx = indices[0][i]
            dist = distances[0][i]
            if 0 <= idx < len(documents):
                results.append((documents[idx], float(dist)))
            else:
                print(f"Warning: Retrieved index {idx} is out of bounds for documents list (size {len(documents)}).")

        return results

    except Exception as e:
        print(f"An error occurred during vector search: {e}")
        return []


# Example Usage (requires embedder_gemini)
if __name__ == "__main__":
    from embedder_gemini import embed_chunks as embed_query # Reuse for query embedding

    # 1. Create dummy data (or use results from previous steps)
    doc1 = Document(page_content="Regional sales show North America leading with $1.2M.")
    doc2 = Document(page_content="Marketing spend focused on paid ads (35%) and content (25%).")
    doc3 = Document(page_content="Future outlook expects growth in LATAM and APAC regions.")
    test_docs = [doc1, doc2, doc3]

    print("--- Running Vector Store Test ---")
    print("Embedding test documents...")
    test_embeddings = embed_query(test_docs) # Use the same embedder

    if test_embeddings:
        # 2. Create and save the vector store
        print("\nCreating vector store...")
        create_vector_store(test_docs, test_embeddings)

        # 3. Load the vector store
        print("\nLoading vector store...")
        loaded_data = load_vector_store()

        if loaded_data:
            loaded_index, loaded_docs = loaded_data

            # 4. Embed a query
            query = "What were the top sales regions?"
            print(f"\nEmbedding query: '{query}'")
            query_emb = embed_query([Document(page_content=query)]) # Embed as a single-item list

            if query_emb:
                # 5. Search the vector store
                print("Searching vector store...")
                search_results = search_vector_store(query_emb[0], loaded_index, loaded_docs, k=2)

                if search_results:
                    print("\nSearch Results:")
                    for doc, score in search_results:
                        print(f"  Score (Distance): {score:.4f}")
                        print(f"  Content: {doc.page_content}")
                else:
                    print("No search results found.")
            else:
                print("Failed to embed query.")
        else:
            print("Failed to load vector store.")

        # Clean up dummy files
        # try:
        #     os.remove(INDEX_FILE)
        #     os.remove(METADATA_FILE)
        #     print("\nCleaned up test files.")
        # except OSError as e:
        #     print(f"\nError cleaning up test files: {e}")

    else:
        print("Failed to generate embeddings for test documents.")