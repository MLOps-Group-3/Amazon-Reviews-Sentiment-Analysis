from google.cloud import aiplatform

def initialize_vertex_ai(project_id, region):
    aiplatform.init(project=project_id, location=region)

def create_dataset(display_name, gcs_source):
    dataset = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=gcs_source
    )
    print(f"Dataset created. Resource name: {dataset.resource_name}")
    return dataset

def main():
    project_id = "amazonreviewssentimentanalysis"
    region = "us-east1"
    
    initialize_vertex_ai(project_id, region)
    
    # GCS path to your data
    gcs_source = "gs://amazon-reviews-sentiment-analysis/pipeline/data/labeled/labeled_data.csv"
    
    # Create a dataset
    dataset = create_dataset(
        display_name="amazon_reviews_sentiment_labeled_data",
        gcs_source=gcs_source
    )

if __name__ == "__main__":
    main()
