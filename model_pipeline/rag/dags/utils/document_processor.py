import os
import logging
import openai
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv("/opt/airflow/.env")
# Removed direct OpenAI API key loading, as we will pass it as a parameter in functions

# Constants
PROMPT_TEMPLATE_PATH = "/opt/airflow/dags/utils/prompt.txt"

# Utility functions
def load_prompt_template(file_path):
    """
    Load the prompt template from a file.

    Parameters:
        file_path (str): Path to the prompt template file.

    Returns:
        str: Contents of the prompt template file.

    Raises:
        FileNotFoundError: If the file is not found at the specified path.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError as e:
        logger.error(f"Prompt template file not found: {e}")
        raise


def split_text(text, max_tokens=12000):
    """
    Split text into manageable chunks based on the token limit.

    Parameters:
        text (str): The text to be split.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        list of str: List of text chunks.

    Logs:
        INFO: Logs the number of chunks the text was split into.
    """
    words = text.split()
    chunks = [
        ' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)
    ]
    logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks


# Main document processing functions
def generate_prompt(text, previous_summary="None"):
    try:
        prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
        prompt = prompt_template.replace("{PREVIOUS_SUMMARY}", previous_summary).replace("{TEXT}", text)
        return prompt
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        raise


def analyze_chunk_with_summary(text, previous_summary="None", api_key=None):
    """
    Analyze a chunk of text with the OpenAI model and return the summary.
    
    Parameters:
        text (str): The text to analyze.
        previous_summary (str): The previous summary to append.
        api_key (str): The OpenAI API key to use for the request.

    Returns:
        str: The analyzed text summary from OpenAI.
    """
    if not api_key:
        logger.error("OpenAI API key is required but not provided.")
        raise ValueError("OpenAI API key is required but not provided.")
    
    openai.api_key = api_key  # Set the provided API key
    
    prompt = generate_prompt(text, previous_summary)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Ensure to use the correct model name here
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please provide responses in plain JSON format without any code block markers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        logger.info("Chunk analyzed successfully.")
        
        # Return the raw output as a string
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e}")
        raise


def process_document_with_summary(document, api_key=None):
    """
    Process the entire document, splitting it into chunks, analyzing each chunk with summaries, and combining them into a final summary.

    Parameters:
        document (dict): The document containing the 'text' key with the content to be processed.
        api_key (str): The OpenAI API key to use for the API call.

    Returns:
        dict: The document with an additional 'analysis' key containing the final summary.
    """
    if not api_key:
        logger.error("OpenAI API key is required but not provided.")
        raise ValueError("OpenAI API key is required but not provided.")
    
    text = document.get("text", "")
    chunks = split_text(text)
    previous_summary = "None"
    all_summaries = []

    for idx, chunk in enumerate(chunks):
        try:
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)}.")
            current_summary = analyze_chunk_with_summary(chunk, previous_summary, api_key)
            all_summaries.append(current_summary)
            previous_summary = current_summary
        except Exception as e:
            logger.error(f"Error processing chunk {idx + 1}: {e}")
            continue

    # Combine all summaries as a single string or list of summaries
    final_summary = "\n".join(all_summaries) if all_summaries else "No summary generated."
    document["analysis"] = final_summary
    logger.info("Document processed successfully.")
    return document
