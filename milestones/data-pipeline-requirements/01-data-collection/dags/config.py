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
    "Appliances",
    "Amazon_Fashion"
]

# Target directory for raw downloads
TARGET_DIRECTORY = "/opt/airflow/data/raw"

# Target directory for processed sample files
TARGET_DIRECTORY_PROCESSED = "/opt/airflow/data/processed"

# Number of worker threads for concurrent downloads
MAX_WORKERS = 6

# Logging directory
LOG_DIRECTORY = "/opt/airflow/logs"

# Sampling fraction for data processing
SAMPLING_FRACTION = 0.01