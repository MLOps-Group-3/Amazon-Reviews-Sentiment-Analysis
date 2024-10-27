from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, split, when

def create_spark_session():
    """
    Create a Spark session.
    Returns:
        SparkSession: Initialized Spark session.
    """
    spark = SparkSession.builder \
        .appName("AmazonReviewDataCleaning") \
        .getOrCreate()
    return spark

def clean_amazon_reviews(spark, df_raw):
    """
    Function to clean Amazon review data at scale using PySpark.
    
    Parameters:
        spark (SparkSession): The Spark session.
        file_path (str): Path to the input CSV file.
    
    Returns:
        DataFrame: Cleaned Spark DataFrame.
    """

    # Read CSV as Spark DataFrame

    # Filter for specific columns
    col_list = ['rating', 'title_x', 'text', 'timestamp', 'helpful_vote', 'verified_purchase', 'categories', 'price', 'parent_asin']
    df_raw = df_raw.select(col_list)

    # 1. Drop duplicates
    df_raw = df_raw.dropDuplicates()

    # 2. Handle missing values
    # - Drop reviews with missing text or rating
    df_raw = df_raw.filter(col("text").isNotNull() & col("rating").isNotNull())    

    # - Replace null lists with empty lists in 'categories'
    df_raw = df_raw.withColumn("categories", split(regexp_replace(col("categories").cast("string"), "null", "[]"), ", "))

    # - Handle missing values by replacing NaN with 'unknown', helpful_vote with 0
    df_raw = df_raw.fillna({"title_x": "unknown", "helpful_vote": 0})
    df_raw = df_raw.withColumn("price", 
                       when(col("price").isNull() | (col("price").cast("string") == "NULL"), "unknown")
                       .otherwise(col("price").cast("string")))
    
    # 3. Clean and normalize text columns
    # - Remove extra whitespaces in 'title_x' and 'text' columns
    df_raw = df_raw.withColumn("title_x", regexp_replace(col("title_x"), r"\s+", " "))
    df_raw = df_raw.withColumn("text", regexp_replace(col("text"), r"\s+", " "))

    return df_raw

if __name__ == "__main__":
    spark = create_spark_session()
    file_path = 'milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/sampled_2018.csv' 
    df_raw = spark.read.csv(file_path, header=True, inferSchema=True, quote='"',escape='"')

    df_cleaned = clean_amazon_reviews(spark, df_raw)

    print("Null values in price after cleaning:", df_cleaned.filter(col("price") == "unknown").count())
    # Show the cleaned data
    df_cleaned.show(truncate=False)
    # Modifying test