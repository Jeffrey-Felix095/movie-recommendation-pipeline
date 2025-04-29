import os, argparse
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import avg, round

# configurate logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def spark_session(appName: str = "calulate_avg_ratings") -> SparkSession:
    spark = SparkSession.builder.appName(appName).getOrCreate()
    return spark


def read_data(spark: SparkSession, file_path: str):
    df = spark.read.parquet(file_path)
    return df


def calculate_avg_ratings(rating_and_movies_df: DataFrame) -> DataFrame:
    ranting_prom_df = rating_and_movies_df.groupBy("movie_id", "title").agg(
        round(avg("rating")).alias("rating_average")
    )
    return ranting_prom_df


def main():
    parser = argparse.ArgumentParser(description="Ratings and movies data")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the directory that contaings ratings and movies join data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to directory where average data will be save",
    )
    args = parser.parse_args()

    spark = spark_session()

    ratings_and_movies_data = os.path.join(
        args.input_path, "ratings_and_movies.parquet"
    )

    avg_ratings_data = os.path.join(args.output_path, "movies_ratings_avg.csv")

    logger.info("Reading ratings_and_movies file...")
    ratings_and_movies_df = read_data(spark, ratings_and_movies_data)

    logger.info("Calculating avg of ratings...")
    movies_ratings_avg_df = calculate_avg_ratings(ratings_and_movies_df)

    logger.info("Saving avg fo ratings file")
    movies_ratings_avg_df.write.mode("overwrite").csv(avg_ratings_data)

    spark.stop()


if __name__ == "__main__":
    main()
