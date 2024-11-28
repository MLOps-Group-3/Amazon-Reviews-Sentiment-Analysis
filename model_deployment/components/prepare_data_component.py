from kfp.v2.dsl import component
from utils.data_loader import load_and_process_data, split_data_by_timestamp

@component(
    base_image="python:3.9-slim",  # Default lightweight Python base image
    packages_to_install=["pandas", "google-cloud-storage", "scikit-learn", "torch"]
)
def prepare_data_component(
    bucket_name: str,
    data_path: str,
    output_dir: str,
) -> str:
    """
    Splits the dataset into train, validation, and test sets and saves them in GCS.

    Args:
        bucket_name (str): Name of the GCS bucket.
        data_path (str): Path to the dataset in GCS.
        output_dir (str): Output directory in GCS for saving split files.

    Returns:
        str: Output directory path in GCS where files are saved.
    """
    import os
    import pickle
    import logging
    from google.cloud import storage
    from utils.data_loader import load_and_process_data, split_data_by_timestamp

    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("prepare_data_component")

    try:
        # Initialize GCS client
        logger.info(f"Initializing GCS client for bucket: {bucket_name}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Check if the file exists in GCS
        blob = bucket.blob(data_path)
        if not blob.exists():
            logger.error(f"The file gs://{bucket_name}/{data_path} does not exist.")
            raise ValueError(f"The file gs://{bucket_name}/{data_path} does not exist.")

        # Download the dataset to a temporary local file
        local_data_path = "/tmp/input.csv"
        logger.info(f"Downloading dataset from gs://{bucket_name}/{data_path} to {local_data_path}")
        blob.download_to_filename(local_data_path)

        # Process the dataset using `load_and_process_data`
        logger.info("Processing dataset...")
        df, label_encoder = load_and_process_data(local_data_path)

        # Split the dataset using `split_data_by_timestamp`
        logger.info("Splitting dataset into train, validation, and test sets...")
        train_df, val_df, test_df = split_data_by_timestamp(df)

        # Save splits locally
        local_output_dir = "/tmp/split_data"
        os.makedirs(local_output_dir, exist_ok=True)
        splits = {"train": train_df, "val": val_df, "test": test_df}
        for split_name, split_df in splits.items():
            file_path = os.path.join(local_output_dir, f"{split_name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(split_df, f)
            logger.info(f"Saved {split_name} data locally at {file_path}")

        # Save the label encoder
        label_encoder_path = os.path.join(local_output_dir, "label_encoder.pkl")
        with open(label_encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)
        logger.info(f"Saved label encoder locally at {label_encoder_path}")

        # Upload splits to GCS
        logger.info(f"Uploading processed data to gs://{bucket_name}/{output_dir}/")
        for file_name in os.listdir(local_output_dir):
            local_file_path = os.path.join(local_output_dir, file_name)
            blob = bucket.blob(f"{output_dir}/{file_name}")
            blob.upload_from_filename(local_file_path)
            logger.info(f"Uploaded {file_name} to gs://{bucket_name}/{output_dir}/")

        return f"Data successfully saved to gs://{bucket_name}/{output_dir}/"

    except Exception as e:
        logger.error(f"Error in prepare_data_component: {str(e)}")
        raise RuntimeError(f"Error in prepare_data_component: {str(e)}")
