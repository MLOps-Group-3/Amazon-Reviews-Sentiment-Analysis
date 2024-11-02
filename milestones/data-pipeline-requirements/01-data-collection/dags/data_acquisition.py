import os
import logging
import datetime
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from airflow.models import Variable
from config import REVIEW_BASE_URL, META_BASE_URL, CATEGORIES, TARGET_DIRECTORY, MAX_WORKERS

def setup_logging():
    log_directory = Variable.get("LOG_DIRECTORY", "/opt/airflow/logs")
    os.makedirs(log_directory, exist_ok=True)
    
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f"data_acquisition_log_{run_id}.log"
    log_file_path = os.path.join(log_directory, log_file_name)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path)
        stream_handler = logging.StreamHandler()
        file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger

logger = setup_logging()

def download_file(url, file_name, chunk_size=8192):
    if os.path.exists(file_name):
        logger.info(f'File {file_name} already exists. Skipping download.')
        return
    try:
        logger.info(f'Starting download for {url}')
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(file_name, 'wb') as file, tqdm(
            desc=os.path.basename(file_name),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = file.write(chunk)
                progress_bar.update(size)
        logger.info(f'Downloaded {file_name}')
    except requests.exceptions.HTTPError as http_err:
        logger.error(f'HTTP error occurred for {url}: {http_err}')
    except Exception as err:
        logger.error(f'An error occurred for {url}: {err}')

def download_category(category, review_base_url, meta_base_url, target_directory):
    category_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'Starting processing for category: {category} at {category_start_time}')
    review_url = f"{review_base_url}{category}.jsonl.gz"
    meta_url = f"{meta_base_url}meta_{category}.jsonl.gz"
    review_file_path = os.path.join(target_directory, f"{category}_reviews.jsonl.gz")
    meta_file_path = os.path.join(target_directory, f"{category}_meta.jsonl.gz")
    download_file(review_url, review_file_path)
    download_file(meta_url, meta_file_path)
    category_end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'Completed processing for category: {category} at {category_end_time}')

def acquire_data():
    logger.info('Download process started.')
    os.makedirs(TARGET_DIRECTORY, exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(lambda cat: download_category(cat, REVIEW_BASE_URL, META_BASE_URL, TARGET_DIRECTORY), CATEGORIES)
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'All download processes completed at {end_time}')

if __name__ == "__main__":
    acquire_data()
