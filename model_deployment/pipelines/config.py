# Project configurations
PROJECT_ID = "amazonreviewssentimentanalysis"
REGION = "us-central1"

# Google Cloud Storage configurations
BUCKET_NAME = "model_deployment_bucket_arsa"
DATA_PATH = "data/labeled_data_1perc.csv"  # Path to the dataset in the bucket
OUTPUT_DIR = "prepared_data"  # Directory in the bucket for output files

# Pipeline configurations
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline"  # Root directory for the pipeline
