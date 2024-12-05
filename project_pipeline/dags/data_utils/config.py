from datetime import datetime, timedelta

# Base URLs for downloading review and metadata data
REVIEW_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
META_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/"

# List of categories to download
CATEGORIES = [
    # Uncomment categories as needed
    # "Home_and_Kitchen",
    # "Clothing_Shoes_and_Jewelry",
    # "Health_and_Household",
    # "Beauty_and_Personal_Care",
    # "Baby_Products",
    # "All_Beauty",
    "Appliances",
    "Amazon_Fashion",
    "Health_and_Personal_Care",
]

# Target directory for downloads
TARGET_DIRECTORY = "/opt/airflow/data/raw"

# Target directory for the sampled files
TARGET_DIRECTORY_SAMPLED = "/opt/airflow/data/sampled"

# Number of worker threads for concurrent downloads
MAX_WORKERS = 6

# Sampling fraction
SAMPLING_FRACTION = 0.01

# Logging directory
LOG_DIRECTORY = "/opt/airflow/logs"

# Directory for dynamic sampled data
SAMPLED_TRAINING_DIRECTORY = "/opt/airflow/data/sampled/training"
SAMPLED_SERVING_DIRECTORY = "/opt/airflow/data/sampled/serving"

TRAINING_SAMPLED_DATA_PATH = "/opt/airflow/data/sampled/training"
SERVING_SAMPLED_DATA_PATH = "/opt/airflow/data/sampled/serving"

# Validation Data Result Path
VALIDATION_RESULT_DATA_PATH = "/opt/airflow/data/validation"

VALIDATION_RESULT_TRAINING_DATA_PATH = "/opt/airflow/data/validation/training"
VALIDATION_RESULT_SERVING_DATA_PATH = "/opt/airflow/data/validation/serving"

# Post preprocessing cleaned data paths
CLEANED_DATA_PATH_TRAIN = "/opt/airflow/data/cleaned/train/cleaned_data.csv"
CLEANED_DATA_PATH_SERVE = "/opt/airflow/data/cleaned/serve/cleaned_data.csv"

# Cleaned aspect data paths
CLEANED_ASPECT_DATA_PATH_TRAIN = "/opt/airflow/data/cleaned/train/aspect_extracted_data.csv"
CLEANED_ASPECT_DATA_PATH_SERVE = "/opt/airflow/data/cleaned/serve/aspect_extracted_data.csv"

# Labeled data paths (only for training)
LABELED_DATA_PATH_TRAIN = "/opt/airflow/data/labeled/train/labeled_data.csv"

# Labeled aspect data paths (only for training)
LABELED_ASPECT_DATA_PATH_TRAIN = "/opt/airflow/data/labeled/train/labeled_aspect_data.csv"

BATCH_PROCESSING_DIRECTORY = "/opt/airflow/data/batch_processing_input"

# Training data configurations
DEFAULT_TRAINING_START_YEAR = 2018
DEFAULT_TRAINING_START_MONTH = 1
TRAINING_PERIOD_MONTHS = 24  # 2 years
TRAINING_SHIFT_MONTHS = 3

# Default values for serving data
DEFAULT_SERVING_YEAR = 2020  # Default year if no files exist
DEFAULT_SERVING_MONTH = 1    # Default month if no files exist
