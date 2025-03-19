# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio",
#     "docling-core",
#     "mlx-vlm",
#     "pillow",
#     "requests",
#     "beautifulsoup4" # Added dependency for BeautifulSoup
# ]
# ///
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
import logging  # Import logging
import os # Import os for path manipulation
import re # Import re for regular expressions
from bs4 import BeautifulSoup # Import BeautifulSoup for XML parsing

import requests
from PIL import Image, UnidentifiedImageError
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, stream_generate
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO) # or logging.DEBUG for more verbose output

## Load the model - Load ONCE outside the function, at the start of the script
model_path = "ds4sd/SmolDocling-256M-preview-mlx-bf16"
logging.info(f"Loading model from path: {model_path}") # Log model loading start
model, processor = load(model_path)
config = load_config(model_path)
logging.info(f"Model loaded successfully.") # Log model loading success


def process_image_to_docling(image_input, prompt_input):
    """
    Processes an image to Docling format using the specified model.

    Args:
        image_input: PIL Image or str (URL or filepath) of the image.
        prompt_input: str prompt for the model.

    Returns:
        tuple: DocTags output (str), Markdown output (str), HTML output (str), Plain Text Output (str).
    """
    ## Settings - These can be exposed as Gradio parameters if needed
    SHOW_IN_BROWSER = False # We don't need to show in browser in Gradio, we will return HTML string

    pil_image = None # Initialize pil_image to None
    image_filepath = None # To store filepath for logging purposes

    ## Prepare input image
    if isinstance(image_input, str): # Assume it's a URL or filepath
        if urlparse(image_input).scheme != "":  # it is a URL
            image_filepath = image_input # URL is the filepath for logging in this case
            try:
                response = requests.get(image_input, stream=True, timeout=10)
                response.raise_for_status()
                pil_image = Image.open(BytesIO(response.content))
            except requests.exceptions.RequestException as e:
                logging.error(f"URL Request Error: {e}, URL: {image_input}") # Log URL request errors with URL
                return f"Error fetching image from URL: {e}", "Error", "Error", "Error" # Handle URL fetch error
            except UnidentifiedImageError as e:
                logging.error(f"UnidentifiedImageError from URL: {e}, URL: {image_input}") # Log UnidentifiedImageError with URL
                return f"Error: Could not identify image from URL: {image_input}. Please check if the URL is a valid image. Details: {e}", "Error", "Error", "Error"
            except Exception as e:
                logging.error(f"Error opening image from URL: {e}, URL: {image_input}") # Log other image opening errors with URL
                return f"Error opening image from URL: {e}", "Error", "Error", "Error" # Handle image opening error
        else: # assume it's a filepath
            image_filepath = image_input
            try:
                pil_image = Image.open(image_input)
            except FileNotFoundError:
                logging.error(f"FileNotFoundError: {image_input}") # Log FileNotFoundError
                return "Error: Image file not found.", "Error", "Error", "Error"
            except UnidentifiedImageError as e:
                logging.error(f"UnidentifiedImageError from File: {e}, Filepath: {image_input}") # Log UnidentifiedImageError with filepath
                return f"Error: Could not identify image file: {image_input}. Please check if the file is a valid image. Details: {e}", "Error", "Error", "Error"
            except Exception as e:
                logging.error(f"Error opening image from filepath: {e}, Filepath: {image_input}") # Log other file opening errors with filepath
                return f"Error opening image from filepath: {e}", "Error", "Error", "Error"
    elif isinstance(image_input, Image.Image): # It's a PIL Image object directly from Gradio upload
        pil_image = image_input
        image_filepath = "<PIL.Image object from upload>" # Indicate it's from upload
    elif image_input is None: # Handle case where no image is input (e.g., user clicks process without uploading)
        return "Error: No image uploaded.", "Error", "Error", "Error"
    else:
        return "Error: Invalid image input type.", "Error", "Error", "Error"

    if pil_image is None: # Should not happen in normal Gradio usage, but for robustness
        return "Error: No image loaded internally.", "Error", "Error", "Error"


    # Apply chat template
    formatted_prompt = apply_chat_template(processor, config, prompt_input, num_images=1)

    ## Generate DocTags output
    doctags_output = ""
    try:
        logging.info(f"Generating DocTags for image: {image_filepath}") # Log generation start with filepath
        for token in stream_generate(
            model, processor, formatted_prompt, [pil_image], max_tokens=4096, verbose=False
        ):
            doctags_output += token.text
            if "</doctag>" in token.text:
                break
        logging.info(f"DocTags generation complete for image: {image_filepath}") # Log generation complete
    except Exception as e:
        logging.error(f"Error during model generation: {e}, Image: {image_filepath}") # Log model generation errors with filepath
        return f"Error during model generation: {e}", "Error", "Error", "Error"


    # Populate document
    try:
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags_output], [pil_image])
        # create a docling document
        doc = DoclingDocument(name="SampleDocument")
        doc.load_from_doctags(doctags_doc)
    except Exception as e:
        logging.error(f"Error processing DocTags output: {e}, Image: {image_filepath}") # Log DocTags processing errors with filepath
        return f"Error processing DocTags output: {e}", "Error", "Error", "Error"

    ## Export as formats
    markdown_output = doc.export_to_markdown()
    html_output = doc.export_to_html(image_mode=ImageRefMode.EMBEDDED)

    # Extract plain text from doctags_output using BeautifulSoup
    soup = BeautifulSoup(doctags_output, 'xml') # Parse as XML
    plain_text_output = soup.get_text(separator='\n', strip=True) # Get text, separated by newlines, and stripped

    return doctags_output, markdown_output, html_output, plain_text_output


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# Docling Image to Document Converter")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image", sources=["upload", "webcam", "clipboard"])
                prompt_input = gr.Textbox(value="Convert this page to docling.", label="Prompt")
                process_button = gr.Button("Process Image")
            with gr.Column():
                doctags_output_box = gr.Code(label="DocTags Output", language="html") # Changed language to "html"
                markdown_output_box = gr.Code(label="Markdown Output", language="markdown")
                html_output_box = gr.Code(label="HTML Output", language="html")
                plain_text_output_box = gr.Code(label="Plain Text Output") # Added Textbox for plain text
                #gr.Markdown("Download button for Plain Text Output requires a newer version of Gradio.") # Informative message


        process_button.click(
            process_image_to_docling,
            inputs=[image_input, prompt_input],
            outputs=[doctags_output_box, markdown_output_box, html_output_box, plain_text_output_box] # Added plain_text_output_box
        )

    demo.launch() # Removed private=True, local access is default. If needed use share=False