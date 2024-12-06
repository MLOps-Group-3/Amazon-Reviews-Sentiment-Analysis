import os
import json
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_metadata_from_documents(documents):
    """Extract metadata from a list of JSON documents and group by category, year, and month."""
    metadata_map = defaultdict(lambda: defaultdict(set))

    for document in documents:
        try:
            # Extract metadata fields
            category = document.get('category', '').strip()
            year = str(document.get('year', '')).strip()
            month = str(document.get('month', '')).strip()

            if category and year and month:
                # Group data by category, year, and month
                metadata_map[category][year].add(month)
            else:
                logging.warning(f"Missing metadata in document: {document}")
        except Exception as e:
            logging.error(f"Error processing document: {e}")

    # Convert sets to lists for JSON serialization
    json_serializable_map = {
        category: {year: list(months) for year, months in years.items()}
        for category, years in metadata_map.items()
    }

    return json_serializable_map

def save_hierarchical_metadata(metadata_map):
    """Save hierarchical metadata to a JSON file."""
    output_folder = '/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/Streamlit/items'
    os.makedirs(output_folder, exist_ok=True)

    hierarchy_file = os.path.join(output_folder, 'hierarchical_metadata.json')
    with open(hierarchy_file, 'w') as f:
        json.dump(metadata_map, f, indent=4)

    logging.info(f"Hierarchical metadata saved to {hierarchy_file}")

# Replace this with the actual path to the refined_process_documents.json file
refined_documents_path = '/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/refined_processed_documents.json'

# Load the refined documents JSON file
try:
    with open(refined_documents_path, 'r') as f:
        refined_process_documents = json.load(f)

    # Extract metadata from documents
    metadata_map = extract_metadata_from_documents(refined_process_documents)

    # Save the hierarchical metadata to a file
    save_hierarchical_metadata(metadata_map)

except FileNotFoundError:
    logging.error(f"File not found: {refined_documents_path}")
except json.JSONDecodeError as e:
    logging.error(f"Error decoding JSON: {e}")