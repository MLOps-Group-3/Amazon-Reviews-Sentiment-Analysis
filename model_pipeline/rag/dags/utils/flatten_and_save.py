import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def flatten_and_save(input_file: str, output_dir: str):
    """
    Flatten and save JSON data into a hierarchical directory structure.

    Args:
        input_file (str): Path to the input JSON file.
        output_dir (str): Base directory to save the chunks.
    """
    try:
        # Load the input JSON file
        with open(input_file, "r") as file:
            data = json.load(file)

        # Ensure data is a list
        if not isinstance(data, list):
            raise ValueError("Input JSON should be a list of records.")

        # Iterate through each record in the list
        for record in data:
            process_record(record, output_dir)

        logging.info("Flattening and saving completed successfully.")
    except Exception as e:
        logging.error("Error during flattening and saving: %s", str(e), exc_info=True)
        raise


def process_record(record: dict, output_dir: str):
    """
    Process a single record and save its chunks in the specified directory.

    Args:
        record (dict): A single record from the JSON data.
        output_dir (str): Base directory to save the chunks.
    """
    try:
        # Extract metadata
        category = record.get("category", "Unknown_Category").strip()
        year = str(record.get("year", "Unknown_Year"))
        month = str(record.get("month", "Unknown_Month"))

        # Define directory path
        base_dir_path = os.path.join(output_dir, category, year, month)

        # Ensure base directory exists
        os.makedirs(base_dir_path, exist_ok=True)

        # Process analysis data
        analysis = record.get("analysis", {})
        product_summaries = analysis.get("product_summaries", {})

        for subcategory, aspects in product_summaries.items():
            # Define subcategory file path
            file_name = f"{subcategory.replace(' ', '_')}.json"
            file_path = os.path.join(base_dir_path, file_name)

            # Ensure all parent directories for the file_path exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save subcategory data to JSON
            with open(file_path, "w") as file:
                json.dump(aspects, file, indent=4)

            logging.info(f"Saved subcategory '{subcategory}' to {file_path}")

    except Exception as e:
        logging.error("Error processing record: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    input_file = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/refined_processed_documents.json"  # Replace with your JSON file path
    output_dir = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/output_chunks"

    # Run the flatten and save process
    flatten_and_save(input_file, output_dir)
