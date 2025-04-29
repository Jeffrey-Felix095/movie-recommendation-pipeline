import os
import argparse
import logging
from typing import Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col,
    regexp_replace,
    regexp_extract,
    explode,
    split,
    trim,
)
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, LongType

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def createSparkSession(app_name: str = "ingest_parse_clean") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def defineSchemas() -> Tuple[StructType, StructType, StructType]:
    ratings_schema = StructType(
        [
            StructField("user_id", IntegerType(), True),
            StructField("movie_id", IntegerType(), True),
            StructField("rating", IntegerType(), True),
            StructField("timestamp", LongType(), True),
        ]
    )
    movie_schema = StructType(
        [
            StructField("movie_id", IntegerType(), True),
            StructField("title", StringType(), True),
            StructField("genres", StringType(), True),
        ]
    )
    user_schema = StructType(
        [
            StructField("user_id", IntegerType(), True),
            StructField("gender", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("ocuppation", IntegerType(), True),
            StructField("zipCode", IntegerType(), True),
        ]
    )
    return ratings_schema, movie_schema, user_schema


def read_and_parse_file(
    spark: SparkSession, file_path: str, schema: StructType, delimiter: str
) -> DataFrame:

    df = spark.read.option("delimiter", delimiter).schema(schema).csv(file_path)

    for field in schema.fields:
        if isinstance(field.dataType, StringType):
            df = df.withColumn(field.name, trim(col(field.name)))

    df = df.dropna(how="all")
    logger.info(f"File read: {file_path}, valid rows: {df.count()}")
    return df


def clean_and_extend_movies_df(movies_df_clean: DataFrame) -> DataFrame:
    """
    Separate years from movie titles and normalize movie genres.
    """
    df = (
        movies_df_clean.withColumn("year", regexp_extract("title", r"\((\d{4})\)", 1))
        .withColumn("title", regexp_replace("title", r"\s*\(\d{4}\)", ""))
        .withColumn("genres", explode(split("genres", r"\|")))
    )
    return df


def main():
    """
    Main function that executes the entire process:
        - Raw data ingestion.
        - Cleansing and transformation.
        - Saving in Parquet format.
    """
    parser = argparse.ArgumentParser(
        description="Ingesta, parseo y limpieza de datos MovieLens."
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        required=True,
        help="Path to MovieLens raw files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save clean files in Parquet format",
    )

    args = parser.parse_args()
    spark = createSparkSession()

    ratings_schema, movies_schema, users_schema = defineSchemas()

    ratings_file = os.path.join(args.raw_path, "ratings.dat")
    movies_file = os.path.join(args.raw_path, "movies.dat")
    users_file = os.path.join(args.raw_path, "users.dat")

    logger.info("Reading and cleaning input raw files...")
    ratings_df_clean = read_and_parse_file(spark, ratings_file, ratings_schema, "::")
    movies_df_clean = read_and_parse_file(spark, movies_file, movies_schema, "::")
    movies_df_clean_and_extend = clean_and_extend_movies_df(movies_df_clean)
    users_df_clean = read_and_parse_file(spark, users_file, users_schema, "::")

    output_ratings_path = os.path.join(args.output_path, "ratings.parquet")
    output_movies_path = os.path.join(args.output_path, "movies.parquet")
    output_users_path = os.path.join(args.output_path, "users.parquet")

    logger.info(f"Saving clean ratings in: {output_ratings_path}")
    ratings_df_clean.write.mode("overwrite").parquet(output_ratings_path)

    logger.info(f"Saving clean películas in: {output_movies_path}")
    movies_df_clean_and_extend.write.mode("overwrite").parquet(output_movies_path)

    logger.info(f"Saving clean usuarios in: {output_users_path}")
    users_df_clean.write.mode("overwrite").parquet(output_users_path)

    logger.info("Processes of ingestion and cleaning finish success.")
    spark.stop()


if __name__ == "__main__":
    main()
