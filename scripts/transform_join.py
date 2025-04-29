import os
import argparse
import logging
from pyspark.sql import SparkSession, DataFrame

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session(appName: str = "transform_join") -> SparkSession:
    return SparkSession.builder.appName(appName).getOrCreate()


def read_file(spark: SparkSession, file_path: str) -> DataFrame:
    return spark.read.parquet(file_path)


def join_rating_and_movie(movies_df: DataFrame, ratings_df: DataFrame) -> DataFrame:
    return movies_df.join(ratings_df, on="movie_id", how="left")


def main():
    """
    Main entry point:
    - Reads the cleaned ratings and movie files.
    - Performs the join.
    - Saves the result as Parquet in the output directory.
    """
    parser = argparse.ArgumentParser(
        description="Transform and join MovieLens clean data"
    )

    parser.add_argument(
        "--clean-data-path",
        type=str,
        required=True,
        help="Path to the directory that contains clean data",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the directory where the join data will be saved",
    )

    args = parser.parse_args()
    spark = create_spark_session()

    clean_ratings_data = os.path.join(args.clean_data_path, "ratings.parquet")
    clean_movies_data = os.path.join(args.clean_data_path, "movies.parquet")

    logger.info("Reading input Parquet files...")
    movies_df = read_file(spark, clean_movies_data)
    ratings_df = read_file(spark, clean_ratings_data)

    logger.info("Uniting ratings and movies...")
    movies_and_rating_df = join_rating_and_movie(movies_df, ratings_df)

    output_file = os.path.join(args.output_path, "ratings_and_movies.parquet")
    logger.info(f"Saving combined result in: {output_file}")
    movies_and_rating_df.write.mode("overwrite").parquet(output_file)

    logger.info("Transformation and union process completed successfully.")
    spark.stop()


if __name__ == "__main__":
    main()
