import logging
import os
import json
import re
from document_processor import process_document_with_summary


def setup_logging():
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
    try:
        with open(file_path, 'w') as f:
            json.dump(documents, f, indent=4)
        logging.info(f"Processed documents saved to '{file_path}'.")
    except Exception as e:
        logging.error(f"Error saving processed documents: {e}")
        raise


def clean_and_validate_json(json_data_list):
    formatted_data = []

    for item in json_data_list:
        try:
            # Step 1: Clean unwanted newline and tab characters
            cleaned_analysis = re.sub(r'\\n|\\t', '', item['analysis'])

            # Step 2: Fix JSON format issues

            # Insert missing commas between dictionaries
            cleaned_analysis = re.sub(r'}\s*{', '},{', cleaned_analysis)

            # Add commas after nested dictionaries if missing before a closing brace or another key
            cleaned_analysis = re.sub(r'(?<=\})(\s*)(?=")', r',\1', cleaned_analysis)

            # Insert commas after Performance section if missing before a closing brace
            cleaned_analysis = re.sub(r'"Performance":\s*{([^{}]*)}\s*(?=[\]}])', r'"Performance": {\1},', cleaned_analysis)

            # Ensure all keys are quoted properly for JSON parsing
            cleaned_analysis = re.sub(r'(?<=\{|,)(\s*)(\w+)(\s*):', r'\1"\2":', cleaned_analysis)

            # Remove any trailing commas at the end of objects or arrays
            cleaned_analysis = re.sub(r',\s*([\]}])', r'\1', cleaned_analysis)

            # Fix extra closing braces at the end
            brace_diff = cleaned_analysis.count("{") - cleaned_analysis.count("}")
            if brace_diff > 0:
                cleaned_analysis += "}" * brace_diff
            elif brace_diff < 0:
                cleaned_analysis = cleaned_analysis[:brace_diff]  # Remove extra closing braces

            # Print cleaned analysis string for inspection (optional)
            print("Cleaned analysis string:", cleaned_analysis)

            # Step 3: Try parsing cleaned string as JSON
            analysis_json = json.loads(cleaned_analysis)

            # Update the item with the properly formatted JSON object for `analysis`
            item['analysis'] = analysis_json
            formatted_data.append(item)

        except json.JSONDecodeError as e:
            # Log parsing issue details
            print(f"Error parsing `analysis` for item with year {item.get('year')}, month {item.get('month')}: {e}")
            print("Failed analysis string:", cleaned_analysis)  # Optional for debugging
            continue

    return formatted_data


def main(documents):
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
    documents_file_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/document_store.json"
    processed_output_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/processed_documents.json"
    refined_output_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/refined_processed_documents.json"

    try:
        # Load, process, and save the initial processed results
        documents = load_documents(documents_file_path)
        processed_results = main(documents)
        save_documents(processed_results, processed_output_path)

        # Load processed results, refine them, and save the refined results
        processed_results = load_documents(processed_output_path)
        refined_results = clean_and_validate_json(processed_results)
        save_documents(refined_results, refined_output_path)

    except Exception as e:
        logging.error(f"An error occurred during document processing: {e}")
