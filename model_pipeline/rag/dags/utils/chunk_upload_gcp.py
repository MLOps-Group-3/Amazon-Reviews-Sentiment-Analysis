from google.cloud import storage
import os

def upload_to_gcs_with_service_account(local_directory, bucket_name, service_account_path, remote_directory):
    """
    Uploads files to a Google Cloud Storage bucket using a service account key file.

    :param local_directory: Path to the local directory containing files to upload.
    :param bucket_name: Name of the GCS bucket.
    :param service_account_path: Path to the service account JSON key file.
    :param remote_directory: Path within the bucket where files will be uploaded.
    """
    try:
        # Set up authentication using the service account key
        client = storage.Client.from_service_account_json(service_account_path)
        bucket = client.get_bucket(bucket_name)

        # Iterate over all files in the local directory
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_directory).replace("\\", "/")  # Handle file paths for all OS
                blob_path = f"{remote_directory}/{relative_path}"  # Remote path in the bucket
                blob = bucket.blob(blob_path)

                # Upload the file
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to {bucket_name}/{blob_path}")

    except Exception as e:
        print(f"Error during GCS upload: {e}")

# Example usage
if __name__ == "__main__":
    local_directory = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/output_chunks"
    gcs_bucket_name = "amazon-reviews-sentiment-analysis"  # Correct bucket name
    service_key_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
    remote_directory = "RAG"  # Remote folder in the bucket

    upload_to_gcs_with_service_account(local_directory, gcs_bucket_name, service_key_path, remote_directory)
