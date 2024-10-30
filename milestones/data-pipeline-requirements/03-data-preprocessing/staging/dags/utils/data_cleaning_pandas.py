import pandas as pd
import re

def clean_amazon_reviews(df_raw):
    """
    Function to clean Amazon review data using Pandas.
    
    Parameters:
        df_raw (DataFrame): Raw DataFrame containing Amazon review data.
    
    Returns:
        DataFrame: Cleaned Pandas DataFrame.
    """

    # 1. Drop duplicates
    df_raw = df_raw.drop_duplicates()

    # 2- Drop reviews with missing text or rating
    df_raw = df_raw.dropna(subset=["text", "rating"])

    # 3. Converting dtype based on range to reduce memory usage
    df_raw['rating'] = df_raw['rating'].astype('int8')
    df_raw["helpful_vote"] = df_raw["helpful_vote"].astype('int32')
    df_raw["average_rating"] = df_raw["average_rating"].astype('float32')
    df_raw["year"] = df_raw["year"].astype('int16')
    df_raw["review_month"] = df_raw["review_month"].astype('int8')

    # 4. Handle missing values

    # - Handle missing values by replacing NaN with 'unknown', helpful_vote with 0
    df_raw["title"].fillna("unknown", inplace=True)
    df_raw["helpful_vote"].fillna(0, inplace=True)
    df_raw["price"] = df_raw["price"].apply(lambda x: "unknown" if pd.isnull(x) or x == "NULL" else str(x))

    # 4. Clean and normalize text columns
    # - Remove extra whitespaces in 'title' and 'text' columns
    df_raw["title"] = df_raw["title"].apply(lambda x: re.sub(r"\s+", " ", x))
    df_raw["text"] = df_raw["text"].apply(lambda x: re.sub(r"\s+", " ", x))

    return df_raw

if __name__ == "__main__":
    # Example usage
    file_path = 'milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/sampled_data_2018_2019.csv'
    df_raw = pd.read_csv(file_path)
    print('df_raw.shape:',df_raw.shape)
    df_cleaned = clean_amazon_reviews(df_raw)
    print('df_cleaned.shape:',df_cleaned.shape)

    # Display the count of rows where price is "unknown" after cleaning
    print("Null values in price after cleaning:", (df_cleaned["price"] == "unknown").sum())
    print(df_cleaned.dtypes)
    # Show a sample of the cleaned data
    print(df_cleaned.head())
    df_cleaned.to_csv('milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/cleaned_data/cleaned_data_2018_2019.csv',index=False)
