# Base URLs for downloading review and metadata data
REVIEW_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
META_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/"

# List of categories to download
CATEGORIES = [
    #"Home_and_Kitchen",
    #"Clothing_Shoes_and_Jewelry",
    #"Health_and_Household",
    #"Beauty_and_Personal_Care",
    "Baby_Products",
    # "All_Beauty",
    # "Appliances",
    # "Amazon_Fashion",
    # "Subscription_Boxes"
]

# Target directory for downloads
TARGET_DIRECTORY = "/opt/airflow/data/raw"

# Target directory for the sampled files
TARGET_DIRECTORY_SAMPLED = "/opt/airflow/data/sampled"

# Number of worker threads for concurrent downloads
MAX_WORKERS = 6

# Logging directory
LOG_DIRECTORY = "/opt/airflow/logs"

# Sampled Data Path
SAMPLED_DATA_PATH = "/opt/airflow/data/sampled/sampled_data_2018_2019.csv"

# Validation Data Result Path
VALIDATION_RESULT_DATA_PATH = "/opt/airflow/data/validation/validation_results.csv"

# Post preprocessing cleaned data path
CLEANED_DATA_PATH = "/opt/airflow/data/cleaned/cleaned_data.csv"

# 
CLEANED_ASPECT_DATA_PATH = "/opt/airflow/data/cleaned/aspect_extracted_data.csv"

LABELED_DATA_PATH = "/opt/airflow/data/labeled/labeled_data.csv"

LABELED_ASPECT_DATA_PATH = "/opt/airflow/data/labeled/labeled_aspect_data.csv"
