# Import necessary functions from review_data_processing.py and embedding_storage.py
from review_data_processing import load_and_process_data, aggregate_data, prepare_documents, save_documents
from embedding_storage import load_documents_from_json, generate_embeddings, initialize_pinecone, upsert_embeddings_to_pinecone

def main():
    # Specify paths for input, output, and Pinecone API information
    input_csv = '/Users/praneethkorukonda/Downloads/Amazon-Reviews-Sentiment-Analysis/data_pipeline/data/labeled/labeled_data.csv'
    output_json = '/Users/praneethkorukonda/Downloads/Amazon-Reviews-Sentiment-Analysis/data_pipeline/data/document_store.json'
    pinecone_api_key = 'YOUR_PINECONE_API_KEY'
    pinecone_index_name = 'amazon-review-summaries'
    embedding_model_name = 'all-MiniLM-L6-v2'
    embedding_dimension = 384

    # Step 1: Load and process review data
    df = load_and_process_data(input_csv)

    # Step 2: Aggregate the data
    aggregated_df = aggregate_data(df)

    # Step 3: Prepare documents for the RAG model
    documents = prepare_documents(aggregated_df)

    # Step 4: Save documents to JSON
    save_documents(documents, output_json)

    # Step 5: Load documents from JSON (for embedding storage)
    documents = load_documents_from_json(output_json)

    # Step 6: Generate embeddings for each document
    documents_with_embeddings = generate_embeddings(documents, embedding_model_name)

    # Step 7: Initialize Pinecone and upsert embeddings
    pinecone_index = initialize_pinecone(pinecone_api_key, pinecone_index_name, embedding_dimension)
    upsert_embeddings_to_pinecone(documents_with_embeddings, pinecone_index)

if __name__ == "__main__":
    main()
