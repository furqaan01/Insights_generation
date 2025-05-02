import streamlit as st
import json
import os
from pathlib import Path
import tempfile
import plotly.express as px
import pandas as pd
from generate_insight_json import extract_text_from_pdf
from insight_generator_gemini import generate_visualization_json, clean_markdown_json
from chunker import chunk_text
from embedder_gemini import embed_chunks
from vector_store import create_vector_store
from context_retriever import ContextRetriever
from query_cli import extract_json_block, build_prompt, call_gemini, save_json_with_backup
from memory_manager import ChatMemory
import copy
import pickle

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
initialize_session()
retriever = ContextRetriever()

# File upload and processing
def process_uploaded_file(uploaded_file):
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()
    temp_path = Path(tempfile.gettempdir()) / filename
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.sidebar.success(f"Uploaded: {filename}")

    if suffix == ".pdf":
        with st.spinner("🔍 Extracting from PDF..."):
            from extractor_vision import extract_content_from_pdf, cleanup_gemini_file
            extracted_text, gemini_file = extract_content_from_pdf(str(temp_path))
            st.session_state.raw_text = extracted_text
            if gemini_file:
                cleanup_gemini_file(gemini_file)
    else:
        st.session_state.raw_text = uploaded_file.read().decode("utf-8")

    if st.session_state.raw_text:
        with st.spinner("✂️ Chunking and embedding..."):
            chunks = chunk_text(st.session_state.raw_text)
            embeddings = embed_chunks(chunks)
            create_vector_store(chunks, embeddings)

        with st.spinner("📊 Generating insight JSON..."):
            raw = generate_visualization_json(st.session_state.raw_text)
            raw_clean = clean_markdown_json(raw)
            result = extract_json_block(raw_clean)
            
            if result and result.get("visualization_possible"):
                st.session_state.json_state = result
                path = filename.replace(suffix, "_charts.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                st.session_state.chart_path = path
                st.session_state.file_processed = True
                st.success("✅ Chart JSON generated and saved.")
            else:
                st.error("No valid charts could be generated from the document.")

# Render charts based on JSON state
def render_charts(json_data):
    if not json_data or not json_data.get("charts"):
        st.warning("No charts available to display.")
        return

    st.subheader("📈 Generated Charts")
    
    for chart in json_data.get("charts", []):
        chart_type = chart.get("chartType")
        title = chart.get("title", chart.get("chartId"))
        x_label = chart.get("xLabel", "X")
        y_label = chart.get("yLabel", "Y")
        data = chart.get("data", {})
        
        st.markdown(f"### {title} ({chart_type})")
        
        try:
            if chart_type == "bar":
                if len(data.get("x", [])) == len(data.get("y", [])):
                    df = pd.DataFrame({x_label: data["x"], y_label: data["y"]})
                    fig = px.bar(df, x=x_label, y=y_label, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif chart_type == "line":
                if len(data.get("x", [])) == len(data.get("y", [])):
                    df = pd.DataFrame({x_label: data["x"], y_label: data["y"]})
                    fig = px.line(df, x=x_label, y=y_label, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif chart_type == "pie":
                if len(data.get("x", [])) == len(data.get("y", [])):
                    fig = px.pie(names=data["x"], values=data["y"], title=title)
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif chart_type == "scatter":
                if len(data.get("x", [])) == len(data.get("y", [])):
                    df = pd.DataFrame({x_label: data["x"], y_label: data["y"]})
                    fig = px.scatter(df, x=x_label, y=y_label, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif chart_type == "heatmap":
                z = data.get("z", [])
                if len(data.get("x", [])) * len(data.get("y", [])) == len(z):
                    z_matrix = [z[i:i+len(data["y"])] for i in range(0, len(z), len(data["y"]))]
                    fig = px.imshow(z_matrix, x=data["x"], y=data["y"], 
                                   labels=dict(x=x_label, y=y_label, color="Value"),
                                   title=title)
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error rendering {chart_type} chart '{title}': {str(e)}")

# Handle user queries and modify charts
def handle_user_query(query, current_json):
    context = retriever.retrieve_context(query) if any(
        k in query.lower() for k in ["explain", "why", "describe"]
    ) else None
    
    prompt = build_prompt(query, current_json, context)
    response = call_gemini(prompt)
    modified = extract_json_block(response)
    
    if modified:
        # Validate the modified JSON preserves the same chart IDs
        old_chart_ids = {c.get("chartId") for c in current_json.get("charts", [])}
        new_chart_ids = {c.get("chartId") for c in modified.get("charts", [])} if modified else set()
        
        if old_chart_ids.issubset(new_chart_ids):
            saved = save_json_with_backup(modified, st.session_state.chart_path)
            if saved:
                st.session_state.memory.add_turn(
                    query,
                    "[Updated]",
                    copy.deepcopy(current_json),
                    modified
                )
                st.session_state.json_state = modified
                st.success("✅ Chart updated as requested.")
                st.rerun()
        else:
            st.warning("⚠️ The modification would change the chart structure. No changes applied.")
    else:
        st.session_state.memory.add_turn(
            query,
            response,
            copy.deepcopy(current_json),
            current_json
        )
        st.info("💡 Gemini Response:")
        st.markdown(response)

# Sidebar file upload
st.sidebar.header("📂 Upload File")
uploaded = st.sidebar.file_uploader("Choose a .pdf or .txt file", type=["pdf", "txt"], 
                                  key="file_uploader")

if uploaded and not st.session_state.file_processed:
    process_uploaded_file(uploaded)

# Main content area
if st.session_state.json_state:
    # Display current JSON configuration
    with st.expander("📄 Current JSON Configuration"):
        st.json(st.session_state.json_state, expanded=False)
    
    # Render all charts
    render_charts(st.session_state.json_state)
    
    # Query interface in sidebar
    with st.sidebar:
        st.subheader("💬 QUERY")
        user_query = st.text_input("What you want ?", 
                                 key="query_input",
                                 placeholder="e.g., 'Change chart1 to line chart'")
        
        if st.button("Apply Changes", key="query_button") and user_query:
            handle_user_query(user_query, st.session_state.json_state)
            
        # Undo functionality
        if st.button("Undo Last Change", key="undo_button"):
            prev_state = st.session_state.memory.get_previous_json_state()
            if prev_state:
                saved = save_json_with_backup(prev_state, st.session_state.chart_path)
                if saved:
                    st.session_state.json_state = prev_state
                    st.success("✅ Reverted to previous state.")
                    st.rerun()
            else:
                st.warning("⚠️ No previous state to revert to.")