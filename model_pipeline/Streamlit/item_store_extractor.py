import os
import json
import logging

def extract_metadata(file_path):
    """Extract metadata from file path."""
    parts = file_path.split('/')
    if len(parts) < 5:
        logging.warning(f"Invalid file path structure: {file_path}")
        return None
    category = parts[1]
    subcategory = parts[4]
    year = parts[2]
    month = parts[3]
    return category, subcategory, year, month

def extract_metadata_from_json_files(folder_path):
    """Extract metadata from all JSON files in a folder."""
    categories = []
    subcategories = []
    years = []
    months = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):  # Process only JSON files
                file_path = os.path.join(root, file)
                
                # Read the JSON file and extract metadata
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        metadata = data.get('metadata', {})
                        if metadata:
                            category = metadata.get('category', '')
                            subcategory = metadata.get('subcategory', '')
                            year = metadata.get('year', '')
                            month = metadata.get('month', '')
                            
                            # Append to the respective lists
                            categories.append(category)
                            subcategories.append(subcategory)
                            years.append(year)
                            months.append(month)
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {e}")
    
    return categories, subcategories, years, months

def save_unique_values_to_text_files(categories, subcategories, years, months):
    """Save unique values to text files in a specified folder."""
    unique_categories = sorted(set(categories))
    unique_subcategories = sorted(set(subcategories))
    unique_years = sorted(set(years))
    unique_months = sorted(set(months))
    
    # Define the folder where text files will be saved
    output_folder = '/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/Streamlit/items'

    # Ensure the folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save each list to a text file in the output folder
    with open(os.path.join(output_folder, 'categories.txt'), 'w') as f:
        f.write("\n".join(unique_categories))
    
    with open(os.path.join(output_folder, 'subcategories.txt'), 'w') as f:
        f.write("\n".join(unique_subcategories))
    
    with open(os.path.join(output_folder, 'years.txt'), 'w') as f:
        f.write("\n".join(unique_years))
    
    with open(os.path.join(output_folder, 'months.txt'), 'w') as f:
        f.write("\n".join(unique_months))

# Folder containing the embedding metadata files
folder_path = '/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/embedding_meta'

# Extract metadata from all files in the folder
categories, subcategories, years, months = extract_metadata_from_json_files(folder_path)

# Save the unique values to text files
save_unique_values_to_text_files(categories, subcategories, years, months)

print("Unique values saved to text files in the specified folder.")
