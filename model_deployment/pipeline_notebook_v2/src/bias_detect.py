import pandas as pd
import logging
import os
from google.cloud import storage

# Configure logging
logging.basicConfig(
    # filename='bias_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def detect_bias(slice_metrics_path):
    # Load slice metrics
    try:
        # Check if path is a GCS URI
        if slice_metrics_path.startswith("gs://"):
            logging.info(f"Detected GCS path: {slice_metrics_path}. Downloading file...")
            
            # Parse GCS path
            bucket_name = slice_metrics_path.split("/")[2]
            blob_path = "/".join(slice_metrics_path.split("/")[3:])
            
            # Download file from GCS
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Save to a temporary local file
            local_file = f"/tmp/{os.path.basename(blob_path)}"
            blob.download_to_filename(local_file)
            logging.info(f"File downloaded from GCS to {local_file}")
            
            # Load the file into a DataFrame
            metrics_df = pd.read_csv(local_file)
        else:
            # Load the file from the local path
            logging.info(f"Loading slice metrics from local path: {slice_metrics_path}")
            metrics_df = pd.read_csv(slice_metrics_path)
    except Exception as e:
        logging.error(f"Failed to load slice metrics: {e}")
        return

    # Get full dataset metrics
    full_dataset_row = metrics_df[metrics_df["Slice Column"] == "Full Dataset"]
    if full_dataset_row.empty:
        logging.error("Full Dataset row not found in metrics. Bias detection cannot proceed.")
        return

    full_samples = int(full_dataset_row["Samples"].iloc[0])
    full_f1 = float(full_dataset_row["F1 Score"].iloc[0])

    # Define thresholds
    min_samples_threshold = 0.1 * full_samples  # Minimum 10% of Full Dataset samples
    f1_threshold = full_f1 * 0.9  # F1 Score less than 90% of Full Dataset F1

    # Filter for potential bias
    biased_rows = metrics_df[
        (metrics_df["Samples"] >= min_samples_threshold) &
        (metrics_df["F1 Score"] < f1_threshold)
    ]


    return biased_rows, f1_threshold
if __name__ == "__main__":
    # Adjust path to slice metrics CSV as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slice_metrics_path = os.path.join(script_dir, "data/slice_metrics.csv")
    detect_bias(slice_metrics_path)
