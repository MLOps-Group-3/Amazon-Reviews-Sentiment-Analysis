import os
import json
import logging
import tempfile
from google.cloud import storage
from dotenv import load_dotenv
import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# GCS and Local Paths
BUCKET_NAME = "amazon-reviews-sentiment-analysis"  # Replace with your GCS bucket name
PREFIX = "RAG/"  # Replace with your GCS folder prefix
OUTPUT_FOLDER = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data"  # Local folder to save embeddings

SERVICE_ACCOUNT_PATH = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"


# Fetch data from GCS bucket
def fetch_data_from_gcs(bucket_name, prefix):
    """
    Fetch JSON data from GCS and return a list of records suitable for embedding.
    """
    try:
        logging.info("Starting to fetch data from GCS...")
        
        # Initialize GCS client
        logging.info("Initializing Google Cloud Storage client.")
        storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_PATH)
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        records = []
        file_paths = []  # List to store the file paths for metadata extraction
        
        for blob in blobs:
            if blob.name.endswith(".json"):
                logging.info(f"Fetching file: {blob.name}")
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    temp_file_path = temp_file.name
                
                try:
                    with open(temp_file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            filtered_data = [item for item in data if isinstance(item, dict)]
                            records.extend(filtered_data)
                            file_paths.extend([blob.name] * len(filtered_data))  # Add corresponding file paths
                            logging.info(f"Processed {len(filtered_data)} dictionary items from {blob.name}.")
                        elif isinstance(data, dict):
                            records.append(data)
                            file_paths.append(blob.name)  # Add corresponding file path
                            logging.info(f"Processed a single dictionary item from {blob.name}.")
                        else:
                            logging.warning(f"Skipping non-dictionary data in file {blob.name}: {type(data)}")
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON in file {blob.name}: {e}")
                finally:
                    os.remove(temp_file_path)

        logging.info(f"Finished fetching data from GCS. Total records fetched: {len(records)}")
        return records, file_paths
    except Exception as e:
        logging.error(f"Error fetching data from GCS: {e}", exc_info=True)
        return [], []


def generate_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None



# Extract metadata from the file path
def extract_metadata(file_path):
    """
    Extract metadata from the file path structure:
    category/year/month/subcategory.json -> Extracts category, subcategory, year, and month.
    """
    parts = file_path.split('/')
    
    # Ensure the parts are sufficient and follow the expected structure
    if len(parts) < 5:
        logging.warning(f"Invalid file path structure: {file_path}")
        return None
    
    # Extract category, subcategory, year, and month from the path
    category = parts[1]  # Category
    subcategory = parts[4]  # Subcategory
    year = parts[2]  # Year
    
    # Handle month extraction logic
    month = parts[3] # First two characters for the month
    
    # # Validate if the month is in correct format (01-12)
    # if month not in [str(i).zfill(2) for i in range(1, 13)]:
    #     logging.warning(f"Invalid or missing month in {file_path}. Setting month to 'Unknown'.")
    #     month = "Unknown"

    return {
        'category': category,
        'subcategory': subcategory,
        'year': year,
        'month': month
    }


# Save embeddings locally with metadata and summaries
def save_embeddings(records, file_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for i, (record, file_path) in enumerate(zip(records, file_paths)):
        try:
            # Extract metadata from the file path
            metadata = extract_metadata(file_path)
            if not metadata:
                continue  # Skip if metadata extraction fails
            
            # Convert the record to a summary (text content for embedding)
            text = json.dumps(record)
            
            # Generate the placeholder embedding (no actual OpenAI call)
            embedding =  generate_embedding(text)
            
            # Construct the output file path
            output_path = os.path.join(output_folder, f"record_{i + 1}_embedding.json")
            
            # Save summary, metadata, and embedding to the output file
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump({
                    'summary': record,  # The entire record as the summary
                    'metadata': metadata,  # Metadata extracted from the folder structure
                    'embedding': embedding  # Placeholder embedding
                }, output_file)
            logging.info(f"Saved embedding for record {i + 1} to {output_path}.")
        except Exception as e:
            logging.error(f"Error saving embedding for record {i + 1}: {e}")


# Main function
def main():
    logging.info("Starting the embedding generation pipeline...")
    
    # Fetch data from GCS
    records, file_paths = fetch_data_from_gcs(BUCKET_NAME, PREFIX)
    if not records:
        logging.warning("No records fetched from GCS. Exiting.")
        return
    
    # Save embeddings with metadata and summary (using placeholder embeddings)
    save_embeddings(records, file_paths, OUTPUT_FOLDER)
    
    logging.info(f"Embedding generation completed. All embeddings saved in '{OUTPUT_FOLDER}'.")


if __name__ == "__main__":
    main()
