import os
import logging

logger = logging.getLogger(__name__)


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
