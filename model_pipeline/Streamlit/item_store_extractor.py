import os
import json
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_metadata(file_path):
    """Extract metadata from a file path."""
    parts = file_path.split('/')
    if len(parts) < 4:
        logging.warning(f"Invalid file path structure: {file_path}")
        return None
    category = parts[1]
    year = parts[2]
    month = parts[3]
    return category, year, month

def extract_metadata_from_json_files(folder_path):
    """Extract metadata from all JSON files in a folder and group by category, year, and month."""
    metadata_map = defaultdict(lambda: defaultdict(set))

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):  # Process only JSON files
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        metadata = data.get('metadata', {})
                        if metadata:
                            category = metadata.get('category', '')
                            year = metadata.get('year', '')
                            month = metadata.get('month', '')

                            if category and year and month:
                                # Group data by category, year, and month
                                metadata_map[category][year].add(month)
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {e}")

    # Convert sets to lists for JSON serialization
    json_serializable_map = {
        category: {year: list(months) for year, months in years.items()}
        for category, years in metadata_map.items()
    }

    return json_serializable_map

def save_hierarchical_metadata(metadata_map):
    """Save hierarchical metadata to a JSON file."""
    output_folder = '/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/Streamlit/items'
    os.makedirs(output_folder, exist_ok=True)

    hierarchy_file = os.path.join(output_folder, 'hierarchical_metadata.json')
    with open(hierarchy_file, 'w') as f:
        json.dump(metadata_map, f, indent=4)

    logging.info(f"Hierarchical metadata saved to {hierarchy_file}")

def get_available_options(metadata_map, category=None, year=None):
    """Get available options dynamically based on the selected hierarchy."""
    if category is None:
        return list(metadata_map.keys())
    if year is None:
        return list(metadata_map.get(category, {}).keys())
    return list(metadata_map.get(category, {}).get(year, []))

# Folder containing the embedding metadata files
folder_path = '/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/embedding_meta'

# Extract metadata from all files in the folder and group by category, year, and month
metadata_map = extract_metadata_from_json_files(folder_path)

# Save the hierarchical metadata to a file
save_hierarchical_metadata(metadata_map)
