import os
import json
from pinecone import Pinecone, ServerlessSpec, Index
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the folder containing your JSON files
json_folder_path = "/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/embedding_meta"

# Initialize Pinecone client by creating an instance of Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the Pinecone index name
index_name = "amazonsentimentanalysis"

# Check if the index exists or create a new one
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' does not exist. Creating a new index...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # Dimension should match the embeddings
        metric="cosine",  # The metric for measuring similarity
        spec=ServerlessSpec(
            cloud="aws",  # Cloud provider
            region="us-east-1"  # AWS region
        )
    )
    print(f"Index '{index_name}' created successfully.")

# Now, use the Index class to interact with the index
index = Index(index_name)  # Use Index directly instead of `pc.index(index_name)`

# Function to prepare upsert data from JSON files
def prepare_upsert_data(json_file_path):
    """
    Prepare upsert data from the JSON file. This function assumes that the JSON
    file contains 'embedding' and 'metadata' fields.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    # Prepare upsert data
    upsert_data = []
    for record in data:
        # Check if the record is a dictionary and contains both 'embedding' and 'metadata'
        if isinstance(record, dict):
            embedding = record.get('embedding', None)
            metadata = record.get('metadata', None)
            doc_id = record.get('doc_id', str(record.get('timestamp', 123456789)))  # Provide default doc_id if none
            
            # Only add the record if embedding and metadata are present
            if embedding and metadata:
                # If the 'embedding' is a dictionary, access its values properly
                if isinstance(embedding, dict):
                    embedding_values = embedding.get('embedding', None)
                    if embedding_values is None:
                        print(f"Skipping record with missing 'embedding' in {json_file_path}")
                        continue
                else:
                    embedding_values = embedding
                
                upsert_data.append({
                    'id': doc_id,
                    'values': embedding_values,  # Embedding values for upsert
                    'metadata': metadata         # Metadata (category, timestamp, etc.)
                })
        else:
            print(f"Skipping record because it's not a dictionary: {record}")
    
    return upsert_data

# Iterate over the JSON files in the folder and upsert the embeddings and metadata into Pinecone
def upsert_embeddings_to_pinecone(json_folder_path):
    for file_name in os.listdir(json_folder_path):
        # Only process JSON files
        if file_name.endswith(".json"):
            file_path = os.path.join(json_folder_path, file_name)
            print(f"\nProcessing file: {file_name}")
            
            # Prepare upsert data from the JSON file
            upsert_data = prepare_upsert_data(file_path)
            
            # Upsert the data into Pinecone if valid data exists
            if upsert_data:
                index.upsert(vectors=upsert_data)
                print(f"Upserted {len(upsert_data)} records from {file_name}")

# Run the upsert process
upsert_embeddings_to_pinecone(json_folder_path)

print("Finished upserting embeddings into Pinecone.")
