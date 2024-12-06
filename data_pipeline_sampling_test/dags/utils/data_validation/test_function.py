import logging
import json
import pandas as pd
from special_characters_detector import check_only_special_characters

# Configure logging to save logs to a file with JSON format
log_file_path = "pipeline_logs.json"
logging.basicConfig(
    filename=log_file_path,
    filemode='a',  # Append to the file instead of overwriting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Removed - %(row)s
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Path to your CSV file
data_path = "/Users/vallimeenaa/Desktop/Group 3 Project/Amazon-Reviews-Sentiment-Analysis/data/sampled_2020_df.csv"

# Load data from CSV file
data = pd.read_csv(data_path)

# Run the checkers and log results
result_1 = check_only_special_characters(data)

# # Log the results
# logger.info(json.dumps({
#     "function": "process_reviews",
#     "result": result_1,
#     "message": "Review processing completed."
# }))

# Uncomment and adjust the following lines for anomaly detection if needed
# result_2 = detect_anomalies(data)
# logger.info(json.dumps({
#     "function": "detect_anomalies",
#     "result": result_2,
#     "message": "Anomaly check completed."
# }))
# print("Anomaly check completed.")
