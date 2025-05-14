import streamlit as st
import json
import os
import re # For robust JSON parsing
from pathlib import Path
import tempfile
import plotly.express as px
import pandas as pd

# from generate_insight_json import extract_text_from_pdf # Not used in this app.py
from insight_generator_gemini import generate_visualization_json
from chunker import chunk_text
from embedder_gemini import embed_chunks
from vector_store import create_vector_store
from context_retriever import ContextRetriever
# extract_json_block is used in handle_user_query, clean_markdown_json is not directly used for initial parsing now
from query_cli import extract_json_block, build_prompt, call_gemini, save_json_with_backup
from memory_manager import ChatMemory
from extractor_vision import extract_content_from_pdf, cleanup_gemini_file # Ensure this is correctly imported

import copy
# import pickle # Pickle is used in query_cli.py for its memory, not directly in app.py state saving here

# --- Robust JSON Parsing Utility ---
def parse_visualization_json_from_raw_text(text: str) -> dict:
    """
    Attempts to extract a valid JSON object for visualization from a string
    that might contain markdown fences, logging lines, or other non-JSON text.
    It prioritizes the last found JSON block.
    """
    json_str = None
    if not isinstance(text, str):
        print(f"⚠️ [parse_visualization_json_from_raw_text] Input is not a string, type: {type(text)}")
        return None

    # 1. Try to find the last markdown-fenced JSON block
    matches = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL))
    if matches:
        json_str = matches[-1].group(1)
        print(f"✅ [JSON Parser] Found JSON in last markdown block.")
    else:
        # 2. If no markdown fences, try to find a JSON object by balancing braces from the last '{'
        print(f"ℹ️ [JSON Parser] No markdown block found. Attempting brace balancing for last JSON object.")
        last_brace_open = text.rfind("{")
        if last_brace_open != -1:
            temp_text_from_last_open_brace = text[last_brace_open:]
            open_braces = 0
            end_index = -1
            for i, char in enumerate(temp_text_from_last_open_brace):
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1
                    if open_braces == 0:
                        end_index = i + 1
                        break
            if end_index != -1:
                json_str = temp_text_from_last_open_brace[:end_index]
                print(f"✅ [JSON Parser] Found JSON by brace balancing from last '{{'.")
            else:
                # Fallback: if matching brace wasn't found cleanly, try first '{' to last '}'
                # This is less reliable if there's garbage at the start/end.
                print(f"⚠️ [JSON Parser] Brace balancing from last '{{' failed. Trying first '{{' to last '}}'.")
                first_brace = text.find("{")
                last_brace = text.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    candidate = text[first_brace : last_brace + 1]
                    # Basic structural check
                    if candidate.count("{") >= candidate.count("}") and candidate.count("[") >= candidate.count("]"): # crude check
                        json_str = candidate
                        print(f"✅ [JSON Parser] Found JSON by first '{{' to last '}}' (less reliable).")
                    else:
                        print(f"⚠️ [JSON Parser] Candidate from first '{{' to last '}}' failed basic structural check.")
                else:
                     print(f"⚠️ [JSON Parser] Could not find any '{{' and '}}' pairs.")
        else:
            print(f"⚠️ [JSON Parser] No '{{' found in text.")

    if not json_str:
        print("❌ [JSON Parser] Could not extract any potential JSON string.")
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"❌ [JSON Parser] Failed to parse extracted JSON string: {e}")
        print(f"Problematic JSON string snippet (approx 500 chars): {json_str[:500]}...")
        return None
# --- End of JSON Parsing Utility ---


# Initialize session state and configure page
def initialize_session():
    if "memory" not in st.session_state:
        st.session_state.memory = ChatMemory()
    if "json_state" not in st.session_state:
        st.session_state.json_state = None
    if "chart_path" not in st.session_state:
        st.session_state.chart_path = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False

st.set_page_config(layout="wide", page_title="📊 Smart Chart Insight Dashboard")
st.title("📊 Smart Chart Insight Dashboard")

# Initialize API key dependent modules after potential .env load
# This assumes API keys are set via .env or globally for all imported modules
# If not, query_cli.py's `genai.configure` might be the one setting it up.
# For robustness, ensure genai.configure is called early, perhaps once here if needed.
# Example:
# from dotenv import load_dotenv
# import google.generativeai as genai
# load_dotenv()
# if os.getenv("GOOGLE_API_KEY"):
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# else:
#     st.error("GOOGLE_API_KEY not found. Please set it in your environment or .env file.")

initialize_session() # Call after genai.configure if it's moved up
retriever = ContextRetriever() # Assumes genai is configured by this point

# File upload and processing
def process_uploaded_file(uploaded_file):
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()
    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / filename
    
    # Reset processing flags for new file
    st.session_state.file_processed = False
    st.session_state.json_state = None
    st.session_state.raw_text = None
    st.session_state.chart_path = None


    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.sidebar.success(f"Uploaded: {filename}")
    gemini_file_for_cleanup = None # To store the file object from vision API for cleanup

    if suffix == ".pdf":
        with st.spinner("🔍 Extracting content from PDF..."):
            try:
                extracted_text_content, gemini_file_for_cleanup = extract_content_from_pdf(str(temp_path))
                if isinstance(extracted_text_content, str) and extracted_text_content.startswith("Error:"):
                    st.error(f"PDF Extraction Failed: {extracted_text_content}")
                    st.session_state.raw_text = None
                elif not extracted_text_content: # Handle empty extraction
                    st.error("PDF Extraction returned no content.")
                    st.session_state.raw_text = None
                else:
                    st.session_state.raw_text = extracted_text_content
                    print(f"✅ [APP.PY] PDF Extracted Text (first 500 chars): {st.session_state.raw_text[:500]}...")
            except Exception as e:
                st.error(f"An error occurred during PDF extraction: {e}")
                st.session_state.raw_text = None
            finally: # Ensure cleanup happens
                if gemini_file_for_cleanup:
                    cleanup_gemini_file(gemini_file_for_cleanup)
    elif suffix == ".txt":
        try:
            st.session_state.raw_text = uploaded_file.read().decode("utf-8")
            print(f"✅ [APP.PY] TXT Extracted Text (first 500 chars): {st.session_state.raw_text[:500]}...")
        except Exception as e:
            st.error(f"Error reading .txt file: {e}")
            st.session_state.raw_text = None
    else:
        st.error(f"Unsupported file type: {suffix}. Please upload a .pdf or .txt file.")
        st.session_state.raw_text = None


    if st.session_state.raw_text:
        with st.spinner("✂️ Chunking text..."):
            chunks = chunk_text(st.session_state.raw_text)
            if not chunks:
                st.warning("Text chunking resulted in no chunks. Skipping embedding and vector store.")
                # Decide if you want to proceed to insight generation without vector store or stop
                # For now, we'll let it try generating insights if raw_text exists.

        if chunks: # Only proceed with embedding if chunks exist
            with st.spinner("🧠 Embedding chunks..."):
                embeddings = embed_chunks(chunks)
                if not embeddings:
                    st.warning("Text embedding failed. Vector store creation will be skipped.")
            
            if embeddings: # Only proceed with vector store if embeddings exist
                with st.spinner("💾 Creating vector store..."):
                    try:
                        # Ensure vector_data directory exists
                        vector_data_dir = "vector_data"
                        os.makedirs(vector_data_dir, exist_ok=True)
                        create_vector_store(chunks, embeddings) # Uses default paths inside vector_data_dir
                        print(f"✅ [APP.PY] Vector store created/updated.")
                    except Exception as e:
                        st.error(f"Failed to create vector store: {e}")

        with st.spinner("📊 Generating insight JSON from Gemini..."):
            raw_gemini_output_str = generate_visualization_json(st.session_state.raw_text)

            print(f"\n--- [APP.PY DEBUG] Raw output from generate_visualization_json ---")
            if isinstance(raw_gemini_output_str, str):
                print(raw_gemini_output_str[:1000] + "..." if len(raw_gemini_output_str) > 1000 else raw_gemini_output_str)
            else:
                print(f"Type of raw_gemini_output_str: {type(raw_gemini_output_str)}")
            print("--- [APP.PY DEBUG] End of raw output ---\n")

            result_dict = None
            if isinstance(raw_gemini_output_str, str) and raw_gemini_output_str.strip():
                result_dict = parse_visualization_json_from_raw_text(raw_gemini_output_str)
            elif isinstance(raw_gemini_output_str, dict): # Should not happen with current generate_visualization_json
                result_dict = raw_gemini_output_str
            
            print(f"--- [APP.PY DEBUG] Result after parse_visualization_json_from_raw_text ---")
            print(result_dict)
            print("--- [APP.PY DEBUG] End of parsed result ---\n")

            if result_dict and result_dict.get("visualization_possible"):
                st.session_state.json_state = result_dict
                # Save with original filename stem in a dedicated output directory
                output_dir = "generated_charts_json"
                os.makedirs(output_dir, exist_ok=True)
                chart_json_filename = Path(filename).stem + "_charts.json"
                st.session_state.chart_path = os.path.join(output_dir, chart_json_filename)
                
                with open(st.session_state.chart_path, "w", encoding="utf-8") as f:
                    json.dump(result_dict, f, indent=2)
                st.session_state.file_processed = True
                st.success(f"✅ Chart JSON generated and saved to {st.session_state.chart_path}")
            else:
                if result_dict and not result_dict.get("visualization_possible"):
                    st.warning("Gemini indicated visualization is not possible for this document.")
                else: # result_dict is None or "visualization_possible" key is missing
                    st.error("No valid charts could be generated or JSON parsing failed for the document.")
                st.session_state.json_state = None
                st.session_state.file_processed = False # Explicitly set to false
    else: # st.session_state.raw_text is None
        st.error("Text extraction failed. Cannot proceed to generate insights.")
        st.session_state.file_processed = False
    
    # Clean up temporary file
    try:
        if temp_path.exists():
            os.remove(temp_path)
            print(f"🧹 Cleaned up temporary file: {temp_path}")
    except Exception as e:
        print(f"⚠️ Error cleaning up temporary file {temp_path}: {e}")


# Render charts based on JSON state
def render_charts(json_data):
    if not json_data or not json_data.get("charts"):
        st.warning("No charts available in the current JSON state to display.")
        return

    st.subheader("📈 Generated Charts")
    
    charts_to_render = json_data.get("charts", [])
    if not charts_to_render:
        st.info("The 'charts' array is empty. Nothing to render.")
        return

    for chart_index, chart in enumerate(charts_to_render):
        if not isinstance(chart, dict):
            st.error(f"Chart object at index {chart_index} is not a valid dictionary. Skipping.")
            continue

        chart_type = chart.get("chartType")
        title = chart.get("title", chart.get("chartId", f"Chart {chart_index + 1}"))
        x_label = chart.get("xLabel", "X-Axis")
        y_label = chart.get("yLabel", "Y-Axis")
        data = chart.get("data", {})
        
        st.markdown(f"#### {title} (Type: {chart_type or 'N/A'})")
        
        if not data or not isinstance(data, dict):
            st.warning(f"No data or invalid data format for chart: {title}. Skipping.")
            continue
        
        x_values = data.get("x", [])
        y_values = data.get("y", [])

        try:
            if chart_type == "bar":
                if x_values and y_values and len(x_values) == len(y_values):
                    df = pd.DataFrame({x_label: x_values, y_label: y_values})
                    fig = px.bar(df, x=x_label, y=y_label, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Data mismatch or missing for bar chart: {title}. X: {len(x_values)}, Y: {len(y_values)}")
                    
            elif chart_type == "line":
                if x_values and y_values and len(x_values) == len(y_values):
                    df = pd.DataFrame({x_label: x_values, y_label: y_values})
                    fig = px.line(df, x=x_label, y=y_label, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Data mismatch or missing for line chart: {title}. X: {len(x_values)}, Y: {len(y_values)}")
                    
            elif chart_type == "pie":
                # Pie charts typically use 'names' (for labels) and 'values'
                # Assuming x_values are names and y_values are values
                if x_values and y_values and len(x_values) == len(y_values):
                    fig = px.pie(names=x_values, values=y_values, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Data mismatch or missing for pie chart: {title}. Names: {len(x_values)}, Values: {len(y_values)}")
                    
            elif chart_type == "scatter":
                if x_values and y_values and len(x_values) == len(y_values):
                    df = pd.DataFrame({x_label: x_values, y_label: y_values})
                    fig = px.scatter(df, x=x_label, y=y_label, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Data mismatch or missing for scatter chart: {title}. X: {len(x_values)}, Y: {len(y_values)}")
                    
            elif chart_type == "heatmap":
                z_values = data.get("z", [])
                # For heatmap, x and y are usually categorical labels for rows/columns
                # z is a 1D array that will be reshaped, or a 2D array directly
                if isinstance(z_values, list) and z_values:
                    if x_values and y_values: # x and y act as labels for the heatmap axes
                        if len(x_values) * len(y_values) == len(z_values): # If z is flat and needs reshaping
                             z_matrix = [z_values[i:i+len(y_values)] for i in range(0, len(z_values), len(y_values))]
                             fig = px.imshow(z_matrix, x=y_values, y=x_values, # Note: px.imshow might expect x=cols, y=rows
                                            labels=dict(x=y_label, y=x_label, color="Value"), title=title)
                             st.plotly_chart(fig, use_container_width=True)
                        elif isinstance(z_values[0], list) and len(z_values) == len(y_values) and len(z_values[0]) == len(x_values): # if z is already 2D
                             fig = px.imshow(z_values, x=x_values, y=y_values,
                                            labels=dict(x=x_label, y=y_label, color="Value"), title=title)
                             st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Data dimensions mismatch for heatmap: {title}. X: {len(x_values)}, Y: {len(y_values)}, Z: {len(z_values)}")
                    else:
                        st.warning(f"Missing X or Y labels for heatmap: {title}.")
                else:
                    st.warning(f"Missing or invalid Z data for heatmap: {title}.")
            else:
                st.info(f"Unsupported or unrecognized chart type '{chart_type}' for chart: {title}")
                    
        except Exception as e:
            st.error(f"Error rendering {chart_type or 'unknown type'} chart '{title}': {str(e)}")
            print(f"❌ [APP.PY RENDER ERROR] Chart: {title}, Type: {chart_type}, Data: {data}, Error: {e}")


# Handle user queries and modify charts
def handle_user_query(query, current_json_state):
    if not current_json_state:
        st.error("Cannot process query: No current chart JSON available.")
        return
    if not st.session_state.chart_path:
        st.error("Cannot process query: Chart path not set (initial JSON might not have been saved).")
        return

    with st.spinner("🧠 Thinking..."):
        context = None
        if any(k_word in query.lower() for k_word in ["explain", "why", "describe", "what does this mean"]):
            print(f"ℹ️ [APP.PY Query] Retrieving context for query: {query}")
            context = retriever.retrieve_context(query) # Assumes vector store is populated

        prompt = build_prompt(query, current_json_state, context) # From query_cli
        response_text = call_gemini(prompt) # From query_cli

        print(f"\n--- [APP.PY DEBUG] Raw Gemini response to user query ---")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print("--- [APP.PY DEBUG] End of raw query response ---\n")

        # extract_json_block is used here for Gemini's response to a modification query
        modified_json_dict = extract_json_block(response_text) # From query_cli

    if modified_json_dict:
        print(f"✅ [APP.PY Query] Gemini proposed JSON modification.")
        # Validate the modified JSON preserves the same chart IDs (or is a valid superset)
        # This validation might need to be more sophisticated depending on allowed modifications
        old_chart_ids = {c.get("chartId") for c in current_json_state.get("charts", [])}
        new_chart_ids = {c.get("chartId") for c in modified_json_dict.get("charts", [])}

        # Allow if new_chart_ids is a superset of old_chart_ids, or if they are the same
        # This allows adding new charts but not removing/renaming existing ones without explicit handling
        # For simplicity, we'll check if all old IDs are present. More complex logic could be added.
        if old_chart_ids.issubset(new_chart_ids):
            saved_successfully = save_json_with_backup(modified_json_dict, st.session_state.chart_path) # From query_cli
            if saved_successfully:
                st.session_state.memory.add_turn(
                    query,
                    "[JSON Updated]", # Placeholder for assistant's textual response if JSON was primary
                    copy.deepcopy(current_json_state),
                    modified_json_dict
                )
                st.session_state.json_state = modified_json_dict
                st.success("✅ Chart configuration updated as per your request.")
                st.rerun()
            else:
                st.error("⚠️ Failed to save the updated JSON configuration.")
        else:
            st.warning("⚠️ The proposed modification would remove or change existing chart IDs, which is not allowed by default. No changes applied.")
            print(f"ℹ️ [APP.PY Query] Modification rejected due to chart ID change. Old: {old_chart_ids}, New: {new_chart_ids}")
    else: # Gemini likely returned an explanation or a message, not a JSON update
        print(f"ℹ️ [APP.PY Query] Gemini returned a textual response (not a JSON modification).")
        st.session_state.memory.add_turn(
            query,
            response_text,
            copy.deepcopy(current_json_state),
            current_json_state # JSON state did not change
        )
        st.info("💡 Gemini's Response:")
        st.markdown(response_text)

# --- UI Layout ---
st.sidebar.header("📂 Upload & Process Document")
uploaded = st.sidebar.file_uploader("Choose a .pdf or .txt file", type=["pdf", "txt"],
                                  key="file_uploader", on_change=lambda: setattr(st.session_state, 'file_processed', False))


if uploaded:
    # If a new file is uploaded, and it hasn't been processed yet in this run, process it.
    # The on_change callback for file_uploader resets file_processed.
    if not st.session_state.get('file_processed', False) or \
       st.session_state.get('last_uploaded_filename') != uploaded.name:
        st.session_state.last_uploaded_filename = uploaded.name # Track the current file
        process_uploaded_file(uploaded)
    elif not st.session_state.json_state: # If processed was true but json_state is None, means prev processing failed
        st.sidebar.warning("Previous processing did not yield charts. Re-upload or try a different file if needed.")


# Main content area: Display charts and query interface if JSON is ready
if st.session_state.get("json_state"):
    # Display current JSON configuration
    with st.expander("📄 Current JSON Configuration", expanded=False):
        st.json(st.session_state.json_state)
    
    render_charts(st.session_state.json_state)
    
    st.sidebar.header("💬 Query & Modify Charts")
    user_query = st.sidebar.text_input("Ask about the charts or request changes:",
                             key="query_input",
                             placeholder="e.g., 'Change chart1 to line chart'")
    
    if st.sidebar.button("Apply Changes / Ask", key="query_button"):
        if user_query:
            handle_user_query(user_query, st.session_state.json_state)
        else:
            st.sidebar.warning("Please enter a query.")
            
    if st.sidebar.button("Undo Last Change", key="undo_button"):
        prev_state = st.session_state.memory.get_previous_json_state()
        if prev_state and st.session_state.chart_path:
            saved = save_json_with_backup(prev_state, st.session_state.chart_path)
            if saved:
                st.session_state.json_state = prev_state
                st.success("✅ Reverted to previous chart configuration.")
                # Remove the last turn from memory as it's effectively undone
                if st.session_state.memory.history:
                    st.session_state.memory.history.pop()
                st.rerun()
            else:
                st.error("Failed to save the reverted state.")
        else:
            st.sidebar.warning("⚠️ No previous chart state to revert to, or chart path is missing.")
else:
    if uploaded and not st.session_state.get('file_processed', False):
        st.info("A file is uploaded. Waiting for processing to complete or for errors to show above.")
    elif not uploaded:
        st.info("👋 Welcome! Please upload a PDF or TXT file using the sidebar to generate and interact with charts.")