import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta  # Ensure this package is installed
from utils.config import TRAINING_SHIFT_MONTHS, TRAINING_PERIOD_MONTHS


def get_next_training_period(directory, category_name, default_start_year=2018, default_start_month=1):
    """
    Determine the next training period for a specific category based on the files in the directory.
    The training period shifts by 3 months with each trigger, and the length of the training period is 2 years.

    Args:
        directory (str): Directory to search for files.
        category_name (str): The name of the category to check.
        default_start_year (int): Default start year if no files exist.
        default_start_month (int): Default start month if no files exist.

    Returns:
        tuple: The start and end dates for the next training period.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    files = os.listdir(directory)
    latest_start_date = None

    for file in files:
        if file.startswith("sampled_data_") and file.endswith(f"{category_name}.csv"):
            parts = file.split("_")
            try:
                # Extract start_date and ensure it's in the correct format
                start_date = datetime.strptime(parts[2], "%Y-%m-%d")
                if latest_start_date is None or start_date > latest_start_date:
                    latest_start_date = start_date
            except (ValueError, IndexError):
                # Skip files with unexpected formats
                continue

    if latest_start_date:
        # Shift the latest start date by 3 months
        new_start_date = latest_start_date + relativedelta(months=TRAINING_SHIFT_MONTHS)
    else:
        # Default to start date if no files exist
        new_start_date = datetime(default_start_year, default_start_month, 1)

    # Calculate the end date as 2 years after the start date, minus one day
    new_end_date = new_start_date + relativedelta(months=TRAINING_PERIOD_MONTHS) - timedelta(days=1)

    return new_start_date.strftime("%Y-%m-%d"), new_end_date.strftime("%Y-%m-%d")
