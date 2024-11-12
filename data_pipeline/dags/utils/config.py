# Base URLs for downloading review and metadata data
REVIEW_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
META_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/"

# List of categories to download
CATEGORIES = [
    "Baby_Products",
    "Appliances",
    "Amazon_Fashion",
    "Health_and_Personal_Care"
]

# Number of worker threads for concurrent downloads
MAX_WORKERS = 6

# Sampling fraction
SAMPLING_FRACTION = 0.1

# GCS paths (to be set in the DAG file)
GCS_BUCKET = "amazon-reviews-sentiment-analysis"
GCS_PIPELINE_PATH = f"gs://{GCS_BUCKET}/pipeline"
GCS_DATA_DIRECTORY = f"{GCS_PIPELINE_PATH}/data"
GCS_LOG_DIRECTORY = f"{GCS_PIPELINE_PATH}/logs"

# GCS paths for different data stages
GCS_RAW_DATA_PATH = f"{GCS_DATA_DIRECTORY}/raw"
GCS_SAMPLED_DATA_PATH = f"{GCS_DATA_DIRECTORY}/sampled/sampled_data_2018_2019.csv"
GCS_VALIDATION_RESULT_DATA_PATH = f"{GCS_DATA_DIRECTORY}/validation/validation_results.csv"
GCS_CLEANED_DATA_PATH = f"{GCS_DATA_DIRECTORY}/cleaned/cleaned_data.csv"
GCS_CLEANED_ASPECT_DATA_PATH = f"{GCS_DATA_DIRECTORY}/cleaned/aspect_extracted_data.csv"
GCS_LABELED_DATA_PATH = f"{GCS_DATA_DIRECTORY}/labeled/labeled_data.csv"
GCS_LABELED_ASPECT_DATA_PATH = f"{GCS_DATA_DIRECTORY}/labeled/labeled_aspect_data.csv"
