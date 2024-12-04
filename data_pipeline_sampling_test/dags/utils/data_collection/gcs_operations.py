from google.cloud import storage
import os
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pull_from_gcs(bucket_name, source_blob_prefix, local_directory, service_account_key):
    """
    Pull data from GCS to local directory.
    
    :param bucket_name: Name of the GCS bucket
    :param source_blob_prefix: Prefix of the blobs to download
    :param local_directory: Local directory to save the files
    :param service_account_key: Path to the service account key file
    """
    logger.info(f"Starting pull operation from GCS bucket '{bucket_name}' with prefix '{source_blob_prefix}'")
    try:
        # Initialize the client with explicit credentials
        client = storage.Client.from_service_account_json(service_account_key)
        logger.info("Successfully initialized GCS client")
        
        # Get the bucket
        bucket = client.get_bucket(bucket_name)
        logger.info(f"Successfully accessed bucket '{bucket_name}'")
        
        # List all blobs with the given prefix
        blobs = bucket.list_blobs(prefix=source_blob_prefix)
        
        # Ensure the local directory exists
        os.makedirs(local_directory, exist_ok=True)
        logger.info(f"Ensured local directory '{local_directory}' exists")
        
        # Download each blob
        download_count = 0
        for blob in blobs:
            # Skip directories
            if blob.name.endswith('/'):
                continue
            
            # Construct the local file path
            local_file_path = os.path.join(local_directory, os.path.basename(blob.name))
            
            # Download the blob
            blob.download_to_filename(local_file_path)
            logger.info(f"Downloaded '{blob.name}' to '{local_file_path}'")
            download_count += 1
        
        logger.info(f"Pull operation completed. Downloaded {download_count} files.")
    except Exception as e:
        logger.error(f"An error occurred during pull operation: {str(e)}")
        raise

def push_to_gcs(bucket_name, source_directory, destination_blob_prefix, service_account_key):
    """
    Push data from local directory to GCS.
    
    :param bucket_name: Name of the GCS bucket
    :param source_directory: Local directory containing files to upload
    :param destination_blob_prefix: Prefix for the destination blobs in GCS
    :param service_account_key: Path to the service account key file
    """
    logger.info(f"Starting push operation to GCS bucket '{bucket_name}' with prefix '{destination_blob_prefix}'")
    try:
        # Initialize the client with explicit credentials
        client = storage.Client.from_service_account_json(service_account_key)
        logger.info("Successfully initialized GCS client")
        
        # Get the bucket
        bucket = client.get_bucket(bucket_name)
        logger.info(f"Successfully accessed bucket '{bucket_name}'")
        
        # List all files in the source directory
        upload_count = 0
        for local_file in glob.glob(os.path.join(source_directory, '*')):
            # Construct the destination blob name
            blob_name = os.path.join(destination_blob_prefix, os.path.basename(local_file))
            
            # Create a blob object
            blob = bucket.blob(blob_name)
            
            # Upload the file
            blob.upload_from_filename(local_file)
            logger.info(f"Uploaded '{local_file}' to '{blob_name}'")
            upload_count += 1
        
        logger.info(f"Push operation completed. Uploaded {upload_count} files.")
    except Exception as e:
        logger.error(f"An error occurred during push operation: {str(e)}")
        raise
