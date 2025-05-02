# embedder_gemini.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
import time
from typing import List, Union

# Load environment variables
load_dotenv()

# Configure the Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# Configuration
EMBEDDING_MODEL = "models/text-embedding-004" # Or other suitable embedding model
BATCH_SIZE = 100  # Max batch size for embedding requests (check API limits)
TASK_TYPE = "RETRIEVAL_DOCUMENT" # For indexing documents for search

def embed_chunks(documents: List[Document]) -> Union[List[List[float]], None]:
    """
    Embeds the content of LangChain Documents using a Gemini embedding model.

    Args:
        documents: A list of LangChain Document objects.

    Returns:
        A list of embeddings (each embedding is a list of floats),
        or None if embedding fails.
    """
    if not documents:
        print("Warning: No documents provided for embedding.")
        return None

    print(f"Embedding {len(documents)} document chunks using model {EMBEDDING_MODEL}...")

    # Extract text content from documents
    texts = [doc.page_content for doc in documents]

    all_embeddings = []
    try:
        # Process texts in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            print(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch_texts)} texts)")

            # Make the embedding request
            # Note: task_type is important for tailoring embeddings
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_texts,
                task_type=TASK_TYPE
            )

            # Check if embeddings were returned
            if "embedding" in result and isinstance(result["embedding"], list):
                all_embeddings.extend(result["embedding"])
            else:
                # Handle potential errors or unexpected format in the result
                print(f"Warning: Unexpected result format or missing embeddings for batch starting at index {i}.")
                print(f"API Result: {result}")
                # Decide if you want to raise an error or just skip the batch
                # For robustness, we might continue and return partial results or None
                # return None # Or raise an exception

            # Simple rate limiting if needed (adjust sleep time)
            # time.sleep(1) # Add a small delay between batches if hitting rate limits

        if len(all_embeddings) == len(texts):
            print(f"Successfully generated {len(all_embeddings)} embeddings.")
            return all_embeddings
        else:
            print(f"Warning: Mismatch in number of embeddings generated ({len(all_embeddings)}) vs expected ({len(texts)}).")
            # Return what was generated, or None, depending on desired behavior
            return all_embeddings if all_embeddings else None

    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        # Consider more specific error handling based on potential API errors
        return None

# Example Usage (for testing this script directly)
if __name__ == "__main__":
    # Create dummy documents
    doc1 = Document(page_content="This is the first document about sales performance.")
    doc2 = Document(page_content="The second document discusses marketing strategies and ROI.")
    doc3 = Document(page_content="Customer feedback and net promoter score trends are analyzed here.")
    test_docs = [doc1, doc2, doc3]

    print("--- Running Embedder Test ---")
    embeddings = embed_chunks(test_docs)

    if embeddings:
        print(f"\nGenerated {len(embeddings)} embeddings.")
        print("Embedding dimension:", len(embeddings[0]))
        # print("First embedding (first 10 values):", embeddings[0][:10])
    else:
        print("\nEmbedding generation failed.")




