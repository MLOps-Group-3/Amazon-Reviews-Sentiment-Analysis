import datetime
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from airflow.models import Variable
from google.cloud import storage
from airflow.utils.log.logging_mixin import LoggingMixin

from ..config import REVIEW_BASE_URL, META_BASE_URL, CATEGORIES, MAX_WORKERS

class GCSDownloader(LoggingMixin):
    def __init__(self):
        super().__init__()

    def download_file(self, url, gcs_file_path, chunk_size=8192):
        client = storage.Client()
        bucket_name, blob_name = gcs_file_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if blob.exists():
            self.log.info(f'File {gcs_file_path} already exists. Skipping download.')
            return

        try:
            self.log.info(f'Starting download for {url}')
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with tqdm(
                desc=blob_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                blob.upload_from_string(response.content, content_type='application/gzip')
                progress_bar.update(total_size)

            self.log.info(f'Downloaded {gcs_file_path}')
        except requests.exceptions.HTTPError as http_err:
            self.log.error(f'HTTP error occurred for {url}: {http_err}')
        except Exception as err:
            self.log.error(f'An error occurred for {url}: {err}')

    def download_category(self, category, review_base_url, meta_base_url, gcs_data_directory):
        category_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log.info(f'Starting processing for category: {category} at {category_start_time}')
        review_url = f"{review_base_url}{category}.jsonl.gz"
        meta_url = f"{meta_base_url}meta_{category}.jsonl.gz"
        review_file_path = f"{gcs_data_directory}/{category}_reviews.jsonl.gz"
        meta_file_path = f"{gcs_data_directory}/{category}_meta.jsonl.gz"
        self.download_file(review_url, review_file_path)
        self.download_file(meta_url, meta_file_path)
        category_end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log.info(f'Completed processing for category: {category} at {category_end_time}')

def acquire_data(gcs_pipeline_path, gcs_data_directory):
    downloader = GCSDownloader()
    downloader.log.info('Download process started.')
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(lambda cat: downloader.download_category(cat, REVIEW_BASE_URL, META_BASE_URL, gcs_data_directory), CATEGORIES)
    
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    downloader.log.info(f'All download processes completed at {end_time}')
