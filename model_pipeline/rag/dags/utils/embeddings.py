import pinecone
import openai
import json
import hashlib
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set API keys from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Index name provided by you
index_name = "amazonsentimentanalysis"

# Check if the index exists; if not, create it
if index_name not in [index['name'] for index in pc.list_indexes()]:
    try:
        # Create an index
        pc.create_index(
            name=index_name,
            dimension=1536,  # Adjust this according to your embedding size
            metric="cosine",  # Use cosine, euclidean, or dotproduct based on your choice
            deletion_protection='disabled',  # Optional, to prevent accidental deletions
            spec=pinecone.ServerlessSpec(
                cloud="aws",  # Define cloud provider
                region="us-east-1"  # Choose your region
            )
        )
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Error during index creation: {e}")

# Connect to the index
index = pc.Index(index_name)

# Function to generate a unique ID
def generate_unique_id(category, month, year, dominant_sentiment):
    """
    Generate a unique ID based on category, month, year, and dominant_sentiment.
    The ID is a hash of these attributes to ensure consistency across runs.
    """
    id_string = f"{category}-{month}-{year}-{dominant_sentiment}"
    return hashlib.sha256(id_string.encode()).hexdigest()

# Function to fetch data from JSON, generate embeddings, and push to Pinecone
def fetch_and_embed_data(json_file_path):
    """
    Fetch data from JSON file, generate embeddings, and push to Pinecone if not already present.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)

    for entry in data:
        # Extract relevant details from the provided JSON
        category = entry.get('category', 'default_category')
        product_name = entry.get('product_name', 'default_product')
        month = entry.get('month', '01')
        year = entry.get('year', '2024')
        text = entry.get('text', '')
        dominant_sentiment = entry.get('dominant_sentiment', 'Neutral')
        analysis = entry.get('analysis', {})  # Extract analysis field

        # Generate unique ID based on category, month, year, and dominant sentiment
        unique_id = generate_unique_id(category, month, year, dominant_sentiment)

        # Check if the ID already exists in Pinecone
        try:
            response = index.fetch(ids=[unique_id])
            if response and response.get('vectors'):
                print(f"ID '{unique_id}' already exists in Pinecone. Skipping this entry.")
                continue  # Skip this entry if the ID already exists
        except Exception as e:
            print(f"Error fetching ID '{unique_id}' from Pinecone: {e}")

        # Generate embeddings for the text
        try:
            embedding = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )['data'][0]['embedding']
            # Ensure embedding is in numpy format
            embedding = np.array(embedding).astype(np.float32)
        except Exception as e:
            print(f"Error generating embedding for ID '{unique_id}': {e}")
            continue

        # Serialize the analysis field to a string (e.g., JSON format)
        analysis_str = json.dumps(analysis)

        # Metadata for Pinecone including serialized analysis
        metadata = {
            "category": category,
            "product_name": product_name,
            "month": month,
            "year": year,
            "dominant_sentiment": dominant_sentiment,
            "analysis": analysis_str  # Store analysis as a JSON string
        }

        # Upsert the embedding to Pinecone with metadata
        try:
            index.upsert(vectors=[{
                'id': unique_id,
                'values': embedding.tolist(),
                'metadata': metadata
            }])
            print(f"Successfully added ID '{unique_id}' to Pinecone with metadata.")
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")

# Run the function with your JSON file path
if __name__ == "__main__":
    json_file_path = "/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/refined_processed_documents_chunk.json"
    fetch_and_embed_data(json_file_path)
