import os
from datetime import datetime

def get_next_serving_month(directory, category_name='', default_year=2020, default_month=1):
    """
    Determine the next serving month for a specific category or concatenated file based on the files in the directory.
    If no files exist, default to January 2020.

    Args:
        directory (str): Directory to search for files.
        category_name (str): The name of the category for which to calculate the next month.
        default_year (int): Default year if no files exist.
        default_month (int): Default month if no files exist.

    Returns:
        tuple: The next year and month for the specified category or concatenated file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    files = os.listdir(directory)
    months = []
    year = default_year

    for file in files:
        # Check both sampled_data_ and concatenated_serving_data_ files
        if ("sampled_data_" in file and category_name in file) or "concatenated_serving_data_" in file:
            parts = file.split("_")
            if len(parts) >= 4 and parts[-2].isdigit() and parts[-1].split('.')[0].isdigit():
                file_year = int(parts[-2])
                file_month = int(parts[-1].split('.')[0])
                months.append((file_year, file_month))

    if months:
        # Find the latest year and month
        latest_year, latest_month = max(months, key=lambda x: (x[0], x[1]))
        next_month = latest_month + 1
        if next_month > 12:
            next_month = 1
            year = latest_year + 1
        else:
            year = latest_year
    else:
        # Default to January 2020 if no files found
        year = default_year
        next_month = default_month

    return year, next_month
