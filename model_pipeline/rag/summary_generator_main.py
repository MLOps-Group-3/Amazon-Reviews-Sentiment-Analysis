import logging
import os
import json
import re
from document_processor import process_document_with_summary


def setup_logging():
    """
    Set up logging configuration for the application.
    Logs are saved in the 'logs' folder as 'processing.log' and output to console.
    """
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_folder, "processing.log")),
            logging.StreamHandler()
        ]
    )


def load_documents(file_path):
    """
    Load documents from a JSON file.

    Parameters:
        file_path (str): Path to the JSON file containing the documents.

    Returns:
        list: List of dictionaries where each dictionary represents a document.
    """
    try:
        with open(file_path, 'r') as f:
            documents = json.load(f)
        logging.info(f"Loaded {len(documents)} documents from {file_path}.")
        return documents
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        raise


def save_documents(documents, file_path):
    """
    Save processed documents to a JSON file.

    Parameters:
        documents (list): List of processed document dictionaries.
        file_path (str): Path where the documents will be saved.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(documents, f, indent=4)
        logging.info(f"Processed documents saved to '{file_path}'.")
    except Exception as e:
        logging.error(f"Error saving processed documents: {e}")
        raise


def attempt_json_fix(raw_text):
    """
    Attempt to fix common JSON issues in the raw text.

    Parameters:
        raw_text (str): The raw JSON-like text from the model output.

    Returns:
        dict or str: Parsed JSON if successful, else original raw text.
    """
    try:
        # Remove code block markers and extra commas
        fixed_text = raw_text.strip("```json").strip("```")
        
        # Replace single quotes with double quotes
        fixed_text = fixed_text.replace("'", '"')
        
        # Remove trailing commas in JSON objects and arrays
        fixed_text = re.sub(r",\s*([\]}])", r"\1", fixed_text)
        
        # Attempt parsing
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        # Log and return the original text if fixing fails
        logging.error("Automatic JSON fixing failed; returning original text.")
        return raw_text


def refine_json(documents):
    """
    Refine the 'analysis' field in each document by parsing JSON.

    Parameters:
        documents (list): List of processed document dictionaries.

    Returns:
        list: Refined list of processed documents.
    """
    refined_documents = []
    
    for i, doc in enumerate(documents):
        refined_doc = doc.copy()
        try:
            # Attempt to parse the analysis field as JSON
            refined_doc["analysis"] = json.loads(doc["analysis"])
            logging.info(f"Successfully parsed 'analysis' field for document {i + 1}.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON in 'analysis' field for document {i + 1}: {e}")
            refined_doc["analysis"] = attempt_json_fix(doc["analysis"])

        refined_documents.append(refined_doc)

    logging.info("Refined JSON structure in all documents.")
    return refined_documents


def main(documents):
    """
    Main function to process a list of documents.

    Parameters:
        documents (list): List of dictionaries where each dictionary contains a document's data.

    Returns:
        list: List of processed documents with cumulative summaries.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    processed_results = []

    for idx, document in enumerate(documents):
        logger.info(f"Processing document {idx + 1}/{len(documents)}.")
        try:
            processed_doc = process_document_with_summary(document)
            processed_results.append(processed_doc)
        except Exception as e:
            logger.error(f"Failed to process document {idx + 1}: {e}")
            continue

    logger.info("All documents processed successfully.")
    return processed_results


if __name__ == "__main__":
    # Path to the JSON file containing the documents
    documents_file_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/document_store.json"
    processed_output_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/processed_documents.json"
    refined_output_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/refined_processed_documents.json"

    try:
        # Load documents
        # documents = load_documents(documents_file_path)
        # processed_results = main(documents)

        # # Save the processed results
        # save_documents(processed_results, processed_output_path)

        # Refine and save the refined results
        processed_results = load_documents(processed_output_path)
        refined_results = refine_json(processed_results)
        save_documents(refined_results, refined_output_path)

    except Exception as e:
        logging.error(f"An error occurred during document processing: {e}")
