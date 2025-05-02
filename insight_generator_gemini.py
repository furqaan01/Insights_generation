# insight_generator_gemini.py
import google.generativeai as genai
import os
import re
from typing import Union

# ✅ HARDCODE your API key here
api_key = "AIzaSyALPsfoJelqx7WfdGVgbyp1J44w8UhjALw"  # Replace with your real API key

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set.")
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-1.5-flash-latest"
GENERATION_TIMEOUT = 300

def clean_markdown_json(text: str) -> str:
    """Strips ```json ... ``` style formatting if it exists"""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text

def generate_visualization_json(context_text: str) -> Union[str, dict]:
    prompt = f"""
You are a data visualization expert. Read the following context and return visualization specifications in the required JSON format.

Context:
{context_text}

Return output in the following schema:

{{
  "visualization_possible": true | false,
  "charts": [
    {{
      "chartId": "unique-id",
      "chartType": "bar|line|pie|scatter|heatmap",
      "title": "...",
      "xLabel": "...",
      "yLabel": "...",
      "data": {{
        "x": [ ... ],
        "y": [ ... ]
      }}
    }}
  ]
}}

Guidelines:
- Ensure that the data for `x` and `y` is appropriate for the chart type (e.g., numeric for bar/line charts, categorical for pie charts).
- If the data is insufficient or unsuitable for generating any charts, return: {{ "visualization_possible": false }}
- If multiple charts are feasible from the data, generate multiple entries within the `charts` array.
- If an error is encountered in processing the data (e.g., missing values, incorrect formatting), return a helpful error message alongside the chart specifications.
- Do not return any text outside the JSON. Return pure JSON only.
"""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            request_options={"timeout": GENERATION_TIMEOUT},
            generation_config=genai.types.GenerationConfig(temperature=0.4)
        )
        print("[🧠] Gemini raw response:")
        print(response.text[:300], "...")

        if not response.text:
            print("[⚠️] Gemini returned an empty response.")

        return response.text

    except Exception as e:
        print(f"[❌] Error in Gemini response: {e}")
        return f"Error: {str(e)}"


