import os
from datetime import datetime, timedelta
import calendar
from utils.config import TRAINING_SHIFT_MONTHS, TRAINING_PERIOD_MONTHS

def get_next_training_period(directory, category_name, default_start_year=2018, default_start_month=1):
    """
    Determine the next training period for a specific category based on the files in the directory.
    If no files exist for the category, default to January 2018 - December 2019.

    Args:
        directory (str): Directory to search for files.
        category_name (str): The name of the category to check.
        default_start_year (int): Default start year if no files exist.
        default_start_month (int): Default start month if no files exist.

    Returns:
        tuple: The start and end dates for the next training period.
    """
    files = os.listdir(directory)
    latest_end_date = None

    for file in files:
        if file.startswith("sampled_data_") and file.endswith(f"{category_name}.csv"):
            parts = file.split("_")
            if len(parts) >= 6:
                end_date = datetime.strptime(parts[4], "%Y-%m-%d")
                if latest_end_date is None or end_date > latest_end_date:
                    latest_end_date = end_date

    if latest_end_date:
        new_start_date = latest_end_date + timedelta(days=1)
        new_start_date += relativedelta(months=TRAINING_SHIFT_MONTHS)
    else:
        new_start_date = datetime(default_start_year, default_start_month, 1)

    new_start_date = new_start_date.replace(day=1)  # Ensure it's the first day of the month
    new_end_date = new_start_date + relativedelta(months=TRAINING_PERIOD_MONTHS) - timedelta(days=1)

    return new_start_date.strftime("%Y-%m-%d"), new_end_date.strftime("%Y-%m-%d")
