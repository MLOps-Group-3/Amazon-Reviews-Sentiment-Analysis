import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(
    # filename='bias_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def detect_bias(slice_metrics_path):
    # Load slice metrics
    try:
        metrics_df = pd.read_csv(slice_metrics_path)
        logging.info(f"Loaded slice metrics from {slice_metrics_path}")
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

    # Log results
    if not biased_rows.empty:
        logging.warning("Potential bias detected in the following slices:")
        for _, row in biased_rows.iterrows():
            logging.warning(
                f"Slice Column: {row['Slice Column']}, Slice Value: {row['Slice Value']}, "
                f"Samples: {row['Samples']}, F1 Score: {row['F1 Score']:.4f} (Threshold: {f1_threshold:.4f})"
            )
        print("Potential bias detected. Check bias_detection.log for details.")
    else:
        logging.info("No significant bias detected.")
        print("No significant bias detected.")

if __name__ == "__main__":
    # Adjust path to slice metrics CSV as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slice_metrics_path = os.path.join(script_dir, "data/slice_metrics.csv")
    detect_bias(slice_metrics_path)
