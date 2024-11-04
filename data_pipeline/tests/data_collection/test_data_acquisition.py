import pytest
import sys
import os
import logging
import requests
from unittest.mock import patch, MagicMock

# Get the path to the data_pipeline directory
data_pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the data_pipeline directory to sys.path
sys.path.insert(0, data_pipeline_dir)

# Now you can import from dags.utils.data_collection
from dags.utils.data_collection.data_acquisition import acquire_data, download_file, download_category, setup_logging

@pytest.fixture
def mock_requests_get():
    with patch('requests.get') as mock_get:
        yield mock_get

@pytest.fixture
def mock_os_makedirs():
    with patch('os.makedirs') as mock_makedirs:
        yield mock_makedirs

@pytest.fixture
def mock_open():
    with patch('builtins.open', create=True) as mock_open:
        yield mock_open

@pytest.fixture
def mock_threadpoolexecutor():
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        yield mock_executor

def test_setup_logging():
    """
    Test that the logging is set up correctly with the expected handlers and log level.
    """
    with patch('os.makedirs') as mock_makedirs:
        logger = setup_logging()
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2
        assert isinstance(logger.handlers[0], logging.FileHandler)
        assert isinstance(logger.handlers[1], logging.StreamHandler)
        mock_makedirs.assert_called_once_with(LOG_DIRECTORY, exist_ok=True)

def test_download_file_success(mock_requests_get, mock_open):
    """
    Test successful file download scenario.
    """
    mock_response = MagicMock()
    mock_response.headers.get.return_value = '1000'
    mock_response.iter_content.return_value = [b'data'] * 10
    mock_requests_get.return_value = mock_response

    download_file('http://example.com/file.gz', 'file.gz')

    mock_requests_get.assert_called_once_with('http://example.com/file.gz', stream=True)
    mock_open.assert_called_once_with('file.gz', 'wb')
    assert mock_open().write.call_count == 10

def test_download_file_http_error(mock_requests_get):
    """
    Test handling of HTTP errors during file download.
    """
    mock_requests_get.side_effect = requests.exceptions.HTTPError('404 Client Error')

    download_file('http://example.com/file.gz', 'file.gz')
    # The function should log the error and not raise an exception

def test_download_category(mock_requests_get, mock_open):
    """
    Test downloading of both review and meta files for a category.
    """
    mock_response = MagicMock()
    mock_response.headers.get.return_value = '1000'
    mock_response.iter_content.return_value = [b'data'] * 10
    mock_requests_get.return_value = mock_response

    download_category('Appliances', REVIEW_BASE_URL, META_BASE_URL, TARGET_DIRECTORY)

    assert mock_requests_get.call_count == 2
    assert mock_open.call_count == 2

@patch('data_acquisition.ThreadPoolExecutor')
@patch('os.makedirs')
def test_acquire_data(mock_makedirs, mock_executor):
    """
    Test the main acquire_data function to ensure it sets up the directory
    and uses ThreadPoolExecutor correctly.
    """
    mock_executor_instance = MagicMock()
    mock_executor.return_value.__enter__.return_value = mock_executor_instance

    acquire_data()

    mock_makedirs.assert_called_once_with(TARGET_DIRECTORY, exist_ok=True)
    mock_executor_instance.map.assert_called_once()

@pytest.mark.parametrize('category', CATEGORIES)
def test_download_category_for_each_category(category, mock_requests_get, mock_open):
    """
    Test downloading files for each category to ensure all categories are processed.
    """
    mock_response = MagicMock()
    mock_response.headers.get.return_value = '1000'
    mock_response.iter_content.return_value = [b'data'] * 10
    mock_requests_get.return_value = mock_response

    download_category(category, REVIEW_BASE_URL, META_BASE_URL, TARGET_DIRECTORY)

    assert mock_requests_get.call_count == 2
    assert mock_open.call_count == 2
