import pinecone
from sentence_transformers import SentenceTransformer
import json

def load_documents_from_json(json_file: str) -> list:
    """
    Loads documents from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing documents.
        
    Returns:
        list: List of documents with text and metadata.
    """
    try:
        with open(json_file, 'r') as f:
            documents = json.load(f)
        return documents
    except Exception as e:
        raise ValueError(f"Error loading documents from JSON: {e}")

def generate_embeddings(documents: list, model_name: str = 'all-MiniLM-L6-v2') -> list:
    """
    Generates embeddings for each document using a specified embedding model.
    
    Args:
        documents (list): List of documents, each containing 'text' field.
        model_name (str): The name of the model used to generate embeddings.
        
    Returns:
        list: List of documents with embeddings added.
    """
    try:
        embedding_model = SentenceTransformer(model_name)
        for doc in documents:
            doc['embedding'] = embedding_model.encode(doc['text']).tolist()
        return documents
    except Exception as e:
        raise ValueError(f"Error generating embeddings: {e}")

def initialize_pinecone(api_key: str, index_name: str, dimension: int = 384) -> pinecone.Index:
    """
    Initializes Pinecone, creates an index if not exists, and connects to it.
    
    Args:
        api_key (str): Pinecone API key.
        index_name (str): Name of the index.
        dimension (int): Dimension of the embeddings.
        
    Returns:
        pinecone.Index: The Pinecone index object.
    """
    pinecone.init(api_key=api_key)
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension, metric="cosine")
    
    return pinecone.Index(index_name)

def upsert_embeddings_to_pinecone(documents: list, index: pinecone.Index) -> None:
    """
    Upserts document embeddings and metadata into Pinecone.
    
    Args:
        documents (list): List of documents with embeddings and metadata.
        index (pinecone.Index): The Pinecone index object.
    """
    try:
        upsert_data = [
            (
                str(i),
                doc['embedding'],
                {
                    'year': doc['year'],
                    'month': doc['month'],
                    'category': doc['category'],
                    'average_rating': doc['average_rating'],
                    'avg_helpful_votes': doc['avg_helpful_votes'],
                    'dominant_sentiment': doc['dominant_sentiment']
                }
            )
            for i, doc in enumerate(documents)
        ]
        index.upsert(upsert_data)
    except Exception as e:
        raise ValueError(f"Error upserting embeddings to Pinecone: {e}")

def main():
    # Define file paths and parameters
    json_file = 'path/to/aggregated_reviews.json'
    api_key = 'YOUR_PINECONE_API_KEY'
    index_name = 'amazon-review-summaries'
    model_name = 'all-MiniLM-L6-v2'
    dimension = 384  # Embedding dimension for the chosen model
    
    # Load documents
    documents = load_documents_from_json(json_file)
    
    # Generate embeddings
    documents = generate_embeddings(documents, model_name)
    
    # Initialize Pinecone index
    index = initialize_pinecone(api_key, index_name, dimension)
    
    # Upsert embeddings to Pinecone
    upsert_embeddings_to_pinecone(documents, index)

if __name__ == "__main__":
    main()
