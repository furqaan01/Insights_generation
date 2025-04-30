from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configuration
CHUNK_SIZE = 1000  # Size of each chunk in characters
CHUNK_OVERLAP = 150  # Overlap between chunks in characters


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """
    Chunks the input text into smaller documents using RecursiveCharacterTextSplitter.

    Args:
        text: The input text string.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The overlap between consecutive chunks.

    Returns:
        A list of LangChain Document objects, each representing a chunk.
    """
    if not text:
        print("Warning: Input text for chunking is empty.")
        return []

    print(f"Chunking text with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,  # Use default separators
        separators=["\n\n", "\n", ". ", " ", ""]  # Sensible separators
    )

    # Create a single Document for splitting, or split directly if splitter handles strings
    # Splitting text directly is often simpler if no complex metadata is needed initially
    chunks = splitter.split_text(text)
    # Convert string chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    print(f"Created {len(documents)} chunks.")
    return documents


# Example Usage (for testing this script directly)
if __name__ == "__main__":
    test_text = """
This is the first paragraph. It contains some sentences.
This is the second paragraph. It also has multiple sentences and provides context.

--- PAGE 2 ---
This marks the beginning of a new section or page. Important details follow here.
Image Description: A bar chart showing monthly sales figures for Q1. X-axis represents months (Jan, Feb, Mar), Y-axis represents sales in USD. Sales increased from $50k in Jan to $75k in Mar.

Another paragraph on page 2 discusses market trends observed during the quarter.
The quick brown fox jumps over the lazy dog. This is filler text to increase length. The quick brown fox jumps over the lazy dog.
"""
    print("--- Running Chunker Test ---")
    docs = chunk_text(test_text)
    if docs:
        print("\n--- First Chunk ---")
        print(docs[0].page_content)
        print(f"\nLength: {len(docs[0].page_content)}")
        if len(docs) > 1:
            print("\n--- Second Chunk ---")
            print(docs[1].page_content)
            print(f"\nLength: {len(docs[1].page_content)}")
            print(f"\nOverlap starts around: {docs[1].page_content[:CHUNK_OVERLAP + 20]}...")