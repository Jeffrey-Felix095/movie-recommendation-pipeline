import os, argparse
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import avg, round, count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def spark_session(appName: str = "calculate_genre_popularity") -> SparkSession:
    spark = SparkSession.builder.appName(appName).getOrCreate()
    return spark


def read_data(spark: SparkSession, file_path: str) -> DataFrame:
    df = spark.read.parquet(file_path)
    return df


def calculate_genre_popularity(ratings_and_movies_df: DataFrame) -> DataFrame:
    genre_popularity_df = ratings_and_movies_df.groupBy("genres").agg(
        count("genres").alias("total movies"),
        round(avg("rating"), 2).alias("average_for_genre"),
    )
    return genre_popularity_df


def main():
    parse = argparse.ArgumentParser(description="Ratings and movies data")
    parse.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the directory that contains ratings and movies data",
    )
    parse.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to directory where data will be saved",
    )

    args = parse.parse_args()

    spark = spark_session()

    ratings_and_movies_data = os.path.join(
        args.input_path, "ratings_and_movies.parquet"
    )

    popularity_genre_data = os.path.join(args.output_path, "popularity_genre.csv")

    logger.info("Reading ratings_and_movies file...")
    ratings_and_movies_df = read_data(spark, ratings_and_movies_data)

    logger.info("Calculating genre popularity...")
    popularity_genre_df = calculate_genre_popularity(ratings_and_movies_df)

    logger.info("Saving genre popularity file..")
    popularity_genre_df.write.mode("overwrite").csv(popularity_genre_data)

    spark.stop()


if __name__ == "__main__":
    main()
