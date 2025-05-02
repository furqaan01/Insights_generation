import argparse
import json
import os
import re

# --- Hardcode working API key here ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCddFnm48fvkpL1l2LeaOF4Tc7mJQss2ps"  # Replace with your working key

from insight_generator_gemini import generate_visualization_json

def extract_text_from_pdf(pdf_path):
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("Please install PyPDF2: pip install PyPDF2")

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def main():
    parser = argparse.ArgumentParser(description="Generate chart insights JSON from a text or PDF file using Gemini.")
    parser.add_argument("filepath", type=str, help="Path to the input .txt or .pdf file")

    args = parser.parse_args()
    file_path = args.filepath

    if not os.path.exists(file_path):
        print("[❌] File does not exist.")
        return

    # --- Read input file ---
    try:
        if file_path.lower().endswith(".pdf"):
            print("[📄] Reading PDF...")
            context_text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                context_text = file.read()
        else:
            print("[❌] Unsupported file format. Use .txt or .pdf")
            return
    except Exception as e:
        print(f"[❌] Error reading file: {e}")
        return

    # --- Generate charts using Gemini ---
    print("[🤖] Generating chart insights using Gemini...")
    raw_result = generate_visualization_json(context_text)

    # --- Clean markdown code block fences if present ---
    if isinstance(raw_result, str):
        raw_result = raw_result.strip()
        if raw_result.startswith("```json"):
            raw_result = re.sub(r"^```json\s*|\s*```$", "", raw_result, flags=re.DOTALL)
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError as e:
            print(f"[❌] JSON parsing failed: {e}")
            print("Raw response from Gemini:")
            print(raw_result[:500] + "...")
            return
    else:
        result = raw_result

    # --- Handle Gemini output ---
    if "error" in result:
        print(f"[❌] Error from Gemini: {result['error']}")
    elif not result.get("visualization_possible"):
        print("[⚠️] Gemini could not generate visualizations.")
    else:
        output_path = os.path.splitext(file_path)[0] + "_charts.json"
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(result, out_file, indent=2)
        print(f"[✅] Chart insights saved to: {output_path}")

if __name__ == "__main__":
    main()