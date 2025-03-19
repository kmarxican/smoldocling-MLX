# SmolDocling-MLX Web Application

This project provides a web application for converting images of documents into Docling format using the `mlx-vlm` model. It leverages `docling-core` for document processing and `gradio` for the user interface.

## Features

*   **Image Input:** Accepts images from URL, file upload, webcam, or clipboard.
*   **Docling Conversion:** Converts images to Docling format using a specified prompt.
*   **Output Formats:** Provides output in DocTags, Markdown, HTML, and plain text formats.
*   **User-Friendly Interface:** Uses Gradio to provide an intuitive web interface.

## Requirements

*   Python 3.12 or higher
*   The dependencies listed in `requirements.txt`

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Create a virtual environment (optional but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```

    Or Conda

    ```bash
    conda create --name smolDoclingMLX python=3.12
    conda activate smolDoclingMLX
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the web application:

    ```bash
    python webapp.py
    ```

2.  Open the provided URL in your web browser.

3.  Upload an image, enter a prompt (optional), and click "Process Image".

4.  View the output in the corresponding boxes.

## Code Structure

*   `webapp.py`: Contains the Gradio web application and the image processing logic.
*   `demo.py`: A demonstration script for converting images to Docling format.
*   `requirements.txt`: Lists the project dependencies.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Dependencies

*   `docling-core`: Core library for Docling document processing.
*   `gradio`: Library for creating the web interface.
*   `mlx-vlm`: MLX model for visual language modeling.
*   `pillow`: Python Imaging Library for image processing.
*   `requests`: Library for making HTTP requests (e.g., fetching images from URLs).
*   `beautifulsoup4`: Library for parsing HTML and XML (used for plain text extraction).

## Model

The application uses the `ds4sd/SmolDocling-256M-preview-mlx-bf16` model.

## Future Plans

*   Add more output formats (e.g., JSON, PDF, LaTeX, etc.).
*   Improve error handling and logging.
*   Enhance the user interface.
*   Add CLI support.
*   Support for multiple images / batch file processing.
*   Build native GUI for desktop.

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

This project is licensed under the AGPL-3.0 license. See the `LICENSE` file for details.

