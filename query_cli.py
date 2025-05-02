import json
import sys
import os
import glob
import google.generativeai as genai
from dotenv import load_dotenv
import re
import copy
import pickle

from memory_manager import ChatMemory
from context_retriever import ContextRetriever

# Load environment variable
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")
genai.configure(api_key=api_key)

# Constants
MEMORY_FILE = "cli_memory.pkl"

# Init or Load memory
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "rb") as f:
        chat_memory = pickle.load(f)
else:
    chat_memory = ChatMemory(max_history_len=10)

retriever = ContextRetriever()

def save_memory():
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(chat_memory, f)

def find_latest_json():
    json_files = [f for f in glob.glob("*.json") if os.path.isfile(f) and not f.endswith("_backup.json")]
    if not json_files:
        print("❌ No JSON files found in the current directory.")
        sys.exit(1)
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"[📂] Using latest modified JSON file: {latest_file}")
    return latest_file

def build_prompt(query: str, json_data: dict, context: str = None) -> str:
    context_note = f"\n\nHere is additional context from the source documents:\n{context}" if context else ""
    return f"""
You are a JSON visualization assistant.

Below is the current chart configuration in JSON format:

```json
{json.dumps(json_data, indent=2)}
```

User query:
{query}
{context_note}

Instructions:
- If the query asks for changes (like chart type, title, color), update only what's requested and return the full updated JSON.
- If the query asks for an explanation, return a plain-text explanation.
- If the request is invalid, say so clearly.
"""

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

def extract_json_block(text: str) -> dict:
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        return None

def save_json_with_backup(data: dict, filepath: str) -> bool:
    if not data.get("visualization_possible") or not data.get("charts"):
        print("⚠️ Chart data is invalid or incomplete. Skipping save.")
        return False
    try:
        backup_path = filepath.replace(".json", "_backup.json")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(filepath, backup_path)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ JSON saved to: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Failed to save JSON: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("❌ Please provide a query.\nExample:")
        print("   python query_cli.py \"Change the chart type to bar\"")
        sys.exit(1)

    user_query = sys.argv[1]
    json_file_path = find_latest_json()

    with open(json_file_path, "r", encoding="utf-8") as f:
        current_json_state = json.load(f)

    # Undo/revert
    if user_query.lower() in ["undo", "revert"]:
        prev_state = chat_memory.get_previous_json_state()
        if prev_state:
            saved = save_json_with_backup(prev_state, json_file_path)
            if saved:
                print("✅ Reverted to previous state:")
                print(json.dumps(prev_state, indent=2))
                chat_memory.add_turn(user_query, "Reverted", current_json_state, prev_state)
                save_memory()
            else:
                print("❌ Failed to revert.")
        else:
            print("⚠️ No previous state to revert to.")
        return

    # Get explanation context if relevant
    needs_explanation = any(word in user_query.lower() for word in ["why", "explain", "reason", "describe"])
    context = retriever.retrieve_context(user_query) if needs_explanation else None

    prompt = build_prompt(user_query, current_json_state, context)
    print("🤖 Sending query to Gemini...")
    response = call_gemini(prompt)
    print("📥 Raw Gemini response:")
    print(response[:300], "..." if len(response) > 300 else "")

    modified = extract_json_block(response)
    if modified:
        saved = save_json_with_backup(modified, json_file_path)
        if saved:
            chat_memory.add_turn(user_query, "[JSON Updated]", copy.deepcopy(current_json_state), modified)
            print(json.dumps(modified, indent=2))
    else:
        chat_memory.add_turn(user_query, response, copy.deepcopy(current_json_state), current_json_state)
        print("📄 Explanation or message from Gemini:\n")
        print(response)

    save_memory()

if __name__ == "__main__":
    main()



