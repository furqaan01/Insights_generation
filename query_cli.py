# query_cli.py
import json
import sys
import os
import glob
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variable
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")
genai.configure(api_key=api_key)

def find_latest_json():
    """Find the most recently modified JSON file in the current directory."""
    json_files = [f for f in glob.glob("*.json") if os.path.isfile(f)]
    if not json_files:
        print("❌ No JSON files found in the current directory.")
        sys.exit(1)
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"[📂] Using latest modified JSON file: {latest_file}")
    return latest_file

def build_prompt(query: str, json_data: dict) -> str:
    """Constructs a prompt for Gemini based on user query and existing JSON."""
    return f"""
You are a JSON visualization assistant.

Below is the current chart configuration in JSON format:

{json.dumps(json_data, indent=2)}

User query:
{query}

Instructions:
- If the query asks for changes (like chart type, title, color), update only what's requested and return the full updated JSON.
- If the query asks for an explanation, return a plain-text explanation.
- If the request is invalid, say so clearly.
"""

def call_gemini(prompt: str) -> str:
    """Calls Gemini with the given prompt."""
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

def is_json(text: str) -> bool:
    """Checks whether a string contains valid JSON."""
    try:
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        json.loads(text)
        return True
    except:
        return False

def extract_json(text: str) -> dict:
    """Extract JSON from Gemini response."""
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def main():
    if len(sys.argv) < 2:
        print("❌ Please provide a query.\nExample:")
        print("   python query_cli.py \"Change the energy intensity trend chart type from line to bar\"")
        sys.exit(1)

    user_query = sys.argv[1]

    # Detect latest JSON file
    json_file_path = find_latest_json()

    # Load JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        original_json = json.load(f)

    # Build prompt and send to Gemini
    prompt = build_prompt(user_query, original_json)
    print("🤖 Sending query to Gemini...")
    response = call_gemini(prompt)

    # Process response
    if is_json(response):
        modified = extract_json(response)

        # Backup original
        #backup_path = json_file_path.replace(".json", "_backup.json")
        #os.rename(json_file_path, backup_path)
        #print(f"[🗂️] Backup saved as: {backup_path}")
        # Backup original
        backup_path = json_file_path.replace(".json", "_backup.json")
        if os.path.exists(backup_path):
            os.remove(backup_path)  # Avoid FileExistsError
        os.rename(json_file_path, backup_path)
        print(f"[🗂️] Backup saved as: {backup_path}")


        # Save new version
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(modified, f, indent=2)

        print("✅ JSON modified and saved successfully.")
        print(json.dumps(modified, indent=2))
    else:
        print("📄 Explanation or message from Gemini:\n")
        print(response)

if __name__ == "__main__":
    main()
