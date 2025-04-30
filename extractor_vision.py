# extractor_vision.py
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv
import mimetypes
import time

# Load environment variables
load_dotenv()

# Configure the Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# Configuration
MODEL_NAME = "gemini-1.5-flash-latest"  # Multimodal model
UPLOAD_TIMEOUT = 300 # Timeout for file upload state check (seconds)
GENERATION_TIMEOUT = 1200 # Timeout for content generation (seconds)

# Helper Function to Upload File to Gemini API
def upload_file_to_gemini(file_path: str, display_name: str = None, timeout: int = UPLOAD_TIMEOUT):
    """Uploads the given file to Gemini and returns the File object."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at path: {file_path}")

    if display_name is None:
        display_name = path.name

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream" # Default fallback
        print(f"Warning: Could not guess MIME type for {file_path}. Using {mime_type}.")
    print(f"Uploading file: {file_path} with MIME type: {mime_type}")

    try:
        start_upload_time = time.time()
        uploaded_file = genai.upload_file(
            path=file_path,
            display_name=display_name,
            mime_type=mime_type
        )
        print(f"Initial upload request sent for {uploaded_file.name}. URI: {uploaded_file.uri}")
        print(f"Waiting for file processing (State: {uploaded_file.state.name})... Max wait: {timeout}s")

        # Wait for the file to become ACTIVE
        while uploaded_file.state.name == "PROCESSING":
            elapsed_time = time.time() - start_upload_time
            if elapsed_time > timeout:
                raise TimeoutError(f"File processing timed out after {timeout} seconds for {uploaded_file.name}")
            print(".", end="", flush=True)
            time.sleep(10) # Check every 10 seconds
            uploaded_file = genai.get_file(name=uploaded_file.name) # Get updated status

        if uploaded_file.state.name == "ACTIVE":
            print(f"\nFile is ACTIVE: {uploaded_file.name}")
            return uploaded_file
        else:
            # Handle FAILED or other non-ACTIVE states
            error_message = f"File processing failed for {uploaded_file.name}. Final state: {uploaded_file.state.name}."
            # Attempt to get specific error if available (structure might vary)
            if hasattr(uploaded_file, 'error') and uploaded_file.error:
                 error_message += f" Reason: {uploaded_file.error}"
            raise Exception(error_message)

    except Exception as e:
        print(f"\nError during file upload/processing for {file_path}: {e}")
        # Attempt to delete if upload started but failed processing
        if 'uploaded_file' in locals() and uploaded_file and uploaded_file.name:
            try:
                print(f"Attempting to delete potentially failed upload: {uploaded_file.name}")
                genai.delete_file(name=uploaded_file.name)
                print("Cleanup successful.")
            except Exception as cleanup_e:
                print(f"Error during file cleanup: {cleanup_e}")
        raise

# Main Extraction Function
def extract_content_from_pdf(file_path: str):
    """
    Extracts text and image descriptions from a PDF using Gemini multimodal model.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A tuple: (extracted_content_string, gemini_file_object) or (error_string, None)
    """
    gemini_file = None
    try:
        print(f"Starting extraction for: {file_path}")
        gemini_file = upload_file_to_gemini(file_path)

        if not gemini_file: # Should not happen if upload_file_to_gemini raises errors correctly
             raise Exception("File upload returned None unexpectedly.")

        model = genai.GenerativeModel(MODEL_NAME)

        prompt = """
Analyze the provided PDF document page by page.

For each page:
1. Extract all textual content accurately. Preserve structure like paragraphs, lists, and tables as text.
2. Identify any images, charts, or graphs.
3. For each visual element, provide a detailed description including:
    - What type of visual it is (photo, bar chart, line graph, map, etc.).
    - What it depicts (e.g., "Bar chart showing sales per region", "Photo of a team meeting").
    - Any visible data points, labels, titles, or legends.
    - Its apparent purpose in the context of the page.

Combine the text and descriptions sequentially for each page. Clearly mark the start of each page, e.g., "--- PAGE X ---".
Output only the extracted text and descriptions. Do not add summaries or interpretations not directly present.
"""

        print(f"Generating content using model {MODEL_NAME} with file: {gemini_file.name}...")
        response = model.generate_content(
            [prompt, gemini_file], # Pass the File object directly
            request_options={"timeout": GENERATION_TIMEOUT}
        )

        print("Content generation finished.")

        # Return the extracted text and the file object (for potential later deletion)
        if hasattr(response, 'text'):
             return response.text, gemini_file
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             block_reason = response.prompt_feedback.block_reason
             block_message = response.prompt_feedback.block_reason_message if hasattr(response.prompt_feedback, 'block_reason_message') else "No details provided."
             error_msg = f"Extraction blocked. Reason: {block_reason}. Details: {block_message}"
             print(error_msg)
             # Still return the gemini_file object for cleanup
             return error_msg, gemini_file
        else:
             print("Warning: Received unexpected response format.")
             print(response) # Log for debugging
             # Still return the gemini_file object for cleanup
             return "Error: Failed to extract content or received unexpected response format.", gemini_file

    except Exception as e:
        print(f"An error occurred during PDF extraction process: {e}")
        # Return error message and the gemini_file if it exists, for cleanup
        return f"Error during extraction: {str(e)}", gemini_file

# Function to clean up the uploaded file
def cleanup_gemini_file(gemini_file):
    """Deletes the uploaded file from Gemini."""
    if gemini_file and gemini_file.name:
        try:
            print(f"Deleting uploaded file from Gemini: {gemini_file.name}")
            genai.delete_file(name=gemini_file.name)
            print(f"Successfully deleted file: {gemini_file.name}")
        except Exception as e:
            print(f"Error deleting file {gemini_file.name}: {e}")
    else:
        print("No Gemini file object provided or file name missing, skipping deletion.")


# Example Usage (for testing this script directly)
if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    # Make sure the test PDF exists relative to this script's location
    test_pdf_path = script_dir / "q1_business_review.pdf" # Adjust if needed

    if test_pdf_path.exists():
        print(f"--- Running Extraction Test on {test_pdf_path} ---")
        extracted_data, uploaded_gemini_file = extract_content_from_pdf(str(test_pdf_path))

        if uploaded_gemini_file and not extracted_data.startswith("Error:"):
            print("\n--- Extracted Content Snippet ---")
            print(extracted_data[:1000] + "...") # Print start of content
            output_file = script_dir / "extracted_output.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(extracted_data)
            print(f"\n--- Full extracted content saved to {output_file} ---")
        else:
            print("\n--- Extraction Failed ---")
            print(extracted_data) # Print the error message

        # Clean up the file on Gemini storage
        cleanup_gemini_file(uploaded_gemini_file)

    else:
        print(f"Test PDF file not found at: {test_pdf_path}")
        print("Please ensure 'q1_business_review.pdf' is in the same directory as extractor_vision.py")


