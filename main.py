'''import argparse
from pathlib import Path
import os
import json
import re

from chunker import chunk_text
from extractor_vision import extract_content_from_pdf, cleanup_gemini_file
from embedder_gemini import embed_chunks
from vector_store import create_vector_store
from insight_generator_gemini import generate_visualization_json
from langchain_core.documents import Document

def clean_markdown_json(raw_text):
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.DOTALL)
    return raw_text

def process_pdf(file_path: str):
    print(f"[📄] Extracting content from: {file_path}")
    extracted_text, gemini_file = extract_content_from_pdf(file_path)

    if not extracted_text or extracted_text.startswith("Error"):
        print(f"[❌] Extraction failed: {extracted_text}")
        return

    print("[✂️] Chunking extracted text...")
    chunks = chunk_text(extracted_text)
    if not chunks:
        print("No chunks created.")
        return

    print(f"[🧠] Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)
    if not embeddings:
        print("Embedding failed.")
        return

    print("[💾] Saving to vector store...")
    output_dir = "vector_data"
    os.makedirs(output_dir, exist_ok=True)
    create_vector_store(
        chunks,
        embeddings,
        index_path=os.path.join(output_dir, "faiss.index"),
        metadata_path=os.path.join(output_dir, "metadata.pkl")
    )

    # ✅ Generate insight JSON
    print("[📊] Generating JSON insight from extracted text...")
    raw_result = generate_visualization_json(extracted_text)

    # Save raw Gemini output for debugging
    raw_text_dump_path = Path(file_path).stem + "_gemini_raw.txt"
    with open(raw_text_dump_path, "w", encoding="utf-8") as f:
        f.write(raw_result if isinstance(raw_result, str) else str(raw_result))
    print(f"[🗃️] Raw Gemini output saved to: {raw_text_dump_path}")

    try:
        if isinstance(raw_result, str):
            raw_result = clean_markdown_json(raw_result)
            insight_data = json.loads(raw_result)
        else:
            insight_data = raw_result

        output_json = Path(file_path).stem + "_charts.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(insight_data, f, indent=2)
        print(f"[✅] Insight JSON saved to: {output_json}")

    except json.JSONDecodeError as e:
        print(f"[❌] JSON parsing failed: {e}")
        print("[⚠️] Raw Gemini response was saved for manual inspection.")

    if gemini_file:
        cleanup_gemini_file(gemini_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF-to-Insight pipeline")
    parser.add_argument("pdf_path", help="Path to PDF file")
    args = parser.parse_args()
    process_pdf(args.pdf_path)
'''

# main.py
import argparse
from pathlib import Path
import os
import re
import json
import json5  # lenient JSON parser

from chunker import chunk_text
from extractor_vision import extract_content_from_pdf, cleanup_gemini_file
from embedder_gemini import embed_chunks
from vector_store import create_vector_store
from insight_generator_gemini import generate_visualization_json
from langchain_core.documents import Document

def clean_markdown_json(raw_text: str) -> str:
    """Removes ```json ... ``` markdown fences."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.DOTALL)
    return raw_text

def extract_first_json_object(raw: str) -> str:
    """Attempts to extract the first valid-looking JSON object from messy output."""
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    return raw[first_brace:last_brace+1] if first_brace != -1 and last_brace != -1 else raw

def process_pdf(file_path: str):
    print(f"[📄] Extracting content from: {file_path}")
    extracted_text, gemini_file = extract_content_from_pdf(file_path)

    if not extracted_text or extracted_text.startswith("Error"):
        print(f"[❌] Extraction failed: {extracted_text}")
        return

    print("[✂️] Chunking extracted text...")
    chunks = chunk_text(extracted_text)
    if not chunks:
        print("No chunks created.")
        return

    print(f"[🧠] Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)
    if not embeddings:
        print("Embedding failed.")
        return

    print("[💾] Saving to vector store...")
    output_dir = "vector_data"
    os.makedirs(output_dir, exist_ok=True)
    create_vector_store(
        chunks,
        embeddings,
        index_path=os.path.join(output_dir, "faiss.index"),
        metadata_path=os.path.join(output_dir, "metadata.pkl")
    )

    # ✅ Generate insight JSON
    print("[📊] Generating JSON insight from extracted text...")
    raw_result = generate_visualization_json(extracted_text)

    # Save raw Gemini response
    raw_path = Path(file_path).stem + "_gemini_raw.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_result if isinstance(raw_result, str) else str(raw_result))
    print(f"[🗃️] Raw Gemini output saved to: {raw_path}")

    try:
        # Clean markdown and extract usable JSON
        if isinstance(raw_result, str):
            raw_clean = clean_markdown_json(raw_result)
            json_like = extract_first_json_object(raw_clean)
            insight_data = json5.loads(json_like)
        else:
            insight_data = raw_result

        # Save parsed JSON
        output_json = Path(file_path).stem + "_charts.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(insight_data, f, indent=2)
        print(f"[✅] Insight JSON saved to: {output_json}")

    except Exception as e:
        print(f"[❌] JSON parsing failed: {e}")
        print("[⚠️] Check the raw Gemini output in:")
        print(raw_path)

    if gemini_file:
        cleanup_gemini_file(gemini_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF-to-Insight pipeline")
    parser.add_argument("pdf_path", help="Path to PDF file")
    args = parser.parse_args()
    process_pdf(args.pdf_path)

