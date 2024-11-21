import json

# Function to flatten a nested JSON object into a flat dictionary excluding 'text' field
def flatten_json(nested_json):
    flat_data = {}

    def flatten(element, parent_key=''):
        if isinstance(element, dict):
            for key, value in element.items():
                if key != "text":  # Exclude the 'text' field
                    flatten(value, f"{parent_key}{key}_")
        elif isinstance(element, list):
            for idx, item in enumerate(element):
                flatten(item, f"{parent_key}{idx}_")
        else:
            flat_data[parent_key[:-1]] = element

    flatten(nested_json)
    return flat_data

# Function to flatten the JSON from the specified file and save to a new file
def flatten_and_save(input_file_path, output_file_path):
    # Load the data from the input JSON file
    with open(input_file_path, 'r') as infile:
        data = json.load(infile)

    # Flatten each record in the JSON data, excluding the 'text' field
    flattened_data = [flatten_json(record) for record in data]

    # Save the flattened data to the output JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(flattened_data, outfile, indent=4)

    print(f"Flattened data saved to {output_file_path}")

# Example usage: Flatten the data and save to a new file
input_file_path = '/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/refined_processed_documents.json'
output_file_path = '/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/flattened_refined_documents.json'

flatten_and_save(input_file_path, output_file_path)
