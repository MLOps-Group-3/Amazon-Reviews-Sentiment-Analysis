# import os
# import json
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import (
#     from_json, schema_of_json, col, from_unixtime, date_format, 
#     month, year, to_timestamp, concat_ws
# )
# import pandas as pd
# from config import TARGET_DIRECTORY, TARGET_DIRECTORY_PROCESSED, SAMPLING_FRACTION, CATEGORIES

# def create_spark_session():
#     os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-21-openjdk-amd64"
#     os.environ['PATH'] = f"{os.path.join(os.environ['JAVA_HOME'], 'bin')}{os.pathsep}{os.environ['PATH']}"
#     os.environ['SPARK_HOME'] = "/home/airflow/.local"
#     os.environ['PATH'] = f"{os.environ['SPARK_HOME']}/bin:{os.environ['PATH']}"

#     print(f"JAVA_HOME set to: {os.environ['JAVA_HOME']}")
#     print(f"SPARK_HOME set to: {os.environ['SPARK_HOME']}")
#     print(f"PATH set to: {os.environ['PATH']}")
    
#     print(f"JAVA_HOME set to: {os.environ['JAVA_HOME']}")

#     # Get the active SparkSession or None if it doesn't exist
#     active_spark = SparkSession.getActiveSession()

#     if active_spark:
#         try:
#             active_spark.stop()
#             print("Stopped old spark session")
#         except Exception as e:
#             print(f"Error stopping Spark session: {e}")

#     try:
#         new_spark = SparkSession.builder \
#             .appName("JsonLoader") \
#             .config("spark.driver.memory", "4g") \
#             .config("spark.local.dir", "/opt/airflow/tmp") \
#             .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
#             .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
#             .getOrCreate()

#         new_spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
        
#         print("New Spark session started.")
#         return new_spark
#     except Exception as e:
#         print(f"Error creating Spark session: {e}")
#         raise

# def load_jsonl_with_spark(spark, jsonl_file):
#     raw_df = spark.read.text(jsonl_file)
#     sample = raw_df.limit(1).collect()[0][0]
#     json_schema = schema_of_json(json.dumps(json.loads(sample)))
#     parsed_df = raw_df.select(from_json(col("value"), json_schema).alias("parsed_data"))
#     flattened_df = parsed_df.select("parsed_data.*")
#     return flattened_df

# def process_reviews_df(reviews_spark_df):
#     reviews_spark_df = reviews_spark_df.withColumn(
#         "review_date_timestamp", 
#         date_format(from_unixtime(col("timestamp") / 1000), "yyyy-MM-dd HH:mm:ss")
#     )
#     filtered_reviews_df = reviews_spark_df.filter(
#         (col("review_date_timestamp").between("2018-01-01 00:00:00", "2020-12-31 23:59:59"))
#     ).drop("images")
#     return filtered_reviews_df

# def join_reviews_and_metadata(filtered_reviews_df, metadata_spark_df):
#     metadata_spark_df_renamed = metadata_spark_df.select(
#         "parent_asin", 
#         "main_category", 
#         col("title").alias("product_name"),
#         "categories", 
#         "price", 
#         "average_rating", 
#         "rating_number"
#     )
#     joined_df = filtered_reviews_df.join(
#         metadata_spark_df_renamed,
#         on="parent_asin",
#         how="left"
#     )
#     return joined_df

# def sample_data(joined_df, sampling_fraction):
#     joined_df = joined_df.withColumn("review_month", month(col("review_date_timestamp")))
#     grouped_df = joined_df.groupBy("review_month", "rating").count()
#     joined_with_count_df = joined_df.join(grouped_df, on=["review_month", "rating"], how="inner")
#     sampled_df = joined_with_count_df.sampleBy(
#         col="review_month",
#         fractions={row['review_month']: sampling_fraction for row in grouped_df.collect()},
#         seed=42
#     )
#     return sampled_df

# def process_sampled_data(sampled_df):
#     sampled_df = sampled_df.withColumn("review_date_timestamp", to_timestamp(sampled_df["review_date_timestamp"], 'yyyy-MM-dd HH:mm:ss'))
#     sampled_df = sampled_df.withColumn("year", year(sampled_df["review_date_timestamp"]))
#     sampled_df = sampled_df.withColumn("categories", concat_ws(",", "categories"))
#     sampled_df = sampled_df.drop("count")
#     return sampled_df

# def split_and_save_data(sampled_df, category_name):
#     sampled_2018_2019_df = sampled_df.filter((sampled_df["year"] >= 2018) & (sampled_df["year"] <= 2019))
#     sampled_2020_df = sampled_df.filter(sampled_df["year"] == 2020)

#     os.makedirs(TARGET_DIRECTORY_PROCESSED, exist_ok=True)

#     pandas_df_2018_2019 = sampled_2018_2019_df.toPandas()
#     pandas_df_2018_2019.to_csv(os.path.join(TARGET_DIRECTORY_PROCESSED, f"{category_name}_2018_2019.csv"), index=False)

#     pandas_df_2020 = sampled_2020_df.toPandas()
#     pandas_df_2020.to_csv(os.path.join(TARGET_DIRECTORY_PROCESSED, f"{category_name}_2020.csv"), index=False)

# def data_sampling(category_name):
#     if category_name not in CATEGORIES:
#         raise ValueError(f"Category '{category_name}' is not in the list of valid categories.")

#     spark = create_spark_session()

#     reviews_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_reviews.jsonl.gz")
#     metadata_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_meta.jsonl.gz")

#     reviews_spark_df = load_jsonl_with_spark(spark, reviews_file)
#     metadata_spark_df = load_jsonl_with_spark(spark, metadata_file)

#     filtered_reviews_df = process_reviews_df(reviews_spark_df)
#     joined_df = join_reviews_and_metadata(filtered_reviews_df, metadata_spark_df)
#     sampled_df = sample_data(joined_df, SAMPLING_FRACTION)
#     processed_sampled_df = process_sampled_data(sampled_df)
#     split_and_save_data(processed_sampled_df, category_name)

#     spark.stop()


import os
import sys
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, schema_of_json, col, from_unixtime, date_format, 
    month, year, to_timestamp, concat_ws
)
from config import TARGET_DIRECTORY, TARGET_DIRECTORY_PROCESSED, SAMPLING_FRACTION, CATEGORIES

def load_jsonl_with_spark(spark, jsonl_file):
    raw_df = spark.read.text(jsonl_file)
    sample = raw_df.limit(1).collect()[0][0]
    json_schema = schema_of_json(json.dumps(json.loads(sample)))
    parsed_df = raw_df.select(from_json(col("value"), json_schema).alias("parsed_data"))
    flattened_df = parsed_df.select("parsed_data.*")
    return flattened_df

def process_reviews_df(reviews_spark_df):
    reviews_spark_df = reviews_spark_df.withColumn(
        "review_date_timestamp", 
        date_format(from_unixtime(col("timestamp") / 1000), "yyyy-MM-dd HH:mm:ss")
    )
    filtered_reviews_df = reviews_spark_df.filter(
        (col("review_date_timestamp").between("2018-01-01 00:00:00", "2020-12-31 23:59:59"))
    ).drop("images")
    return filtered_reviews_df

def join_reviews_and_metadata(filtered_reviews_df, metadata_spark_df):
    metadata_spark_df_renamed = metadata_spark_df.select(
        "parent_asin", 
        "main_category", 
        col("title").alias("product_name"),
        "categories", 
        "price", 
        "average_rating", 
        "rating_number"
    )
    joined_df = filtered_reviews_df.join(
        metadata_spark_df_renamed,
        on="parent_asin",
        how="left"
    )
    return joined_df

def sample_data(joined_df, sampling_fraction):
    joined_df = joined_df.withColumn("review_month", month(col("review_date_timestamp")))
    grouped_df = joined_df.groupBy("review_month", "rating").count()
    joined_with_count_df = joined_df.join(grouped_df, on=["review_month", "rating"], how="inner")
    sampled_df = joined_with_count_df.sampleBy(
        col="review_month",
        fractions={row['review_month']: sampling_fraction for row in grouped_df.collect()},
        seed=42
    )
    return sampled_df

def process_sampled_data(sampled_df):
    sampled_df = sampled_df.withColumn("review_date_timestamp", to_timestamp(sampled_df["review_date_timestamp"], 'yyyy-MM-dd HH:mm:ss'))
    sampled_df = sampled_df.withColumn("year", year(sampled_df["review_date_timestamp"]))
    sampled_df = sampled_df.withColumn("categories", concat_ws(",", "categories"))
    sampled_df = sampled_df.drop("count")
    return sampled_df

def split_and_save_data(sampled_df, category_name):
    sampled_2018_2019_df = sampled_df.filter((sampled_df["year"] >= 2018) & (sampled_df["year"] <= 2019))
    sampled_2020_df = sampled_df.filter(sampled_df["year"] == 2020)

    os.makedirs(TARGET_DIRECTORY_PROCESSED, exist_ok=True)

    pandas_df_2018_2019 = sampled_2018_2019_df.toPandas()
    pandas_df_2018_2019.to_csv(os.path.join(TARGET_DIRECTORY_PROCESSED, f"{category_name}_2018_2019.csv"), index=False)

    pandas_df_2020 = sampled_2020_df.toPandas()
    pandas_df_2020.to_csv(os.path.join(TARGET_DIRECTORY_PROCESSED, f"{category_name}_2020.csv"), index=False)

if __name__ == "__main__":
    # Parse command-line argument for category_name
    if len(sys.argv) != 2:
        raise ValueError("Please provide the category name as an argument.")
    
    category_name = sys.argv[1]
    if category_name not in CATEGORIES:
        raise ValueError(f"Category '{category_name}' is not in the list of valid categories.")
    
    # Initialize Spark session
    spark = SparkSession.builder.appName("DataSamplingJob").getOrCreate()

    # Load, process, sample, and save data
    reviews_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_reviews.jsonl.gz")
    metadata_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_meta.jsonl.gz")

    reviews_spark_df = load_jsonl_with_spark(spark, reviews_file)
    metadata_spark_df = load_jsonl_with_spark(spark, metadata_file)

    filtered_reviews_df = process_reviews_df(reviews_spark_df)
    joined_df = join_reviews_and_metadata(filtered_reviews_df, metadata_spark_df)
    sampled_df = sample_data(joined_df, SAMPLING_FRACTION)
    processed_sampled_df = process_sampled_data(sampled_df)
    split_and_save_data(processed_sampled_df, category_name)

    spark.stop()
