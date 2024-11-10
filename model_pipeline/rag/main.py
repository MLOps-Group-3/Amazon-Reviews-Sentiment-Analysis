# Import necessary functions from process_reviews.py
from review_data_processing import load_and_process_data, aggregate_data, prepare_documents, save_documents

def main():
    # Specify the paths for input and output
    input_csv = '/Users/praneethkorukonda/Downloads/Amazon-Reviews-Sentiment-Analysis/data_pipeline/data/labeled/labeled_data.csv'  # Path to your input CSV file
    output_json = '/Users/praneethkorukonda/Downloads/Amazon-Reviews-Sentiment-Analysis/data_pipeline/data/document_store.json'  # Path to the output JSON file

    # Load and process data
    df = load_and_process_data(input_csv)

    # Aggregate the data
    aggregated_df = aggregate_data(df)

    # Prepare documents for RAG model
    documents = prepare_documents(aggregated_df)

    # Save the documents to a JSON file
    save_documents(documents, output_json)

if __name__ == "__main__":
    # Run the main function
    main()
