import argparse, os
import logging
from pyspark.sql import SparkSession, DataFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "prepare_als_data") -> SparkSession:
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark


def read_ratings_data(spark: SparkSession, file_path: str) -> DataFrame:
    """Read the clean ratings data."""
    logger.info(f"Reading ratings data from: {file_path}")
    df = spark.read.parquet(file_path)
    return df


def split_data(
    ratings_df: DataFrame, train_ratio: float, validation_ratio: float, seed: int
) -> tuple[DataFrame, DataFrame]:
    """Divide the data into training and validation sets."""
    logger.info(
        f"Splitting data into training ({train_ratio}) and validation ({validation_ratio}) sets with seed {seed}"
    )

    (training_data, validation_data) = ratings_df.randomSplit(
        [train_ratio, validation_ratio], seed=seed
    )

    logger.info(f"Training data count: {training_data.count()}")
    logger.info(f"Validation data count: {validation_data.count()}")
    return training_data, validation_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for ALS model training (split into train/validation)."
    )
    parser.add_argument(
        "--input-ratings-path",
        type=str,
        required=True,
        help="Path to the cleaned ratings Parquet data.",
    )
    parser.add_argument(
        "--output-train-path",
        type=str,
        required=True,
        help="Path to save the training data Parquet file.",
    )
    parser.add_argument(
        "--output-validation-path",
        type=str,
        required=True,
        help="Path to save the validation data Parquet file.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,  
        help="Random seed for data splitting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,  
        help="Proportion of data for the training set.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,  
        help="Proportion of data for the validation set.",
    )

    args = parser.parse_args()

    spark = create_spark_session()

    logger.info("Reading clean ratings file...")
    ratings_df = read_ratings_data(spark, args.input_ratings_path)

    logger.info("Spliting the data...")
    training_data, validation_data = split_data(
        ratings_df, args.train_ratio, args.validation_ratio, args.split_seed
    )

    logger.info(f"Saving training data to: {args.output_train_path}")
    training_data.write.mode("overwrite").parquet(args.output_train_path)

    logger.info(f"Saving validation data to: {args.output_validation_path}")
    validation_data.write.mode("overwrite").parquet(args.output_validation_path)

    logger.info("Data preparation for ALS completed successfully.")

    spark.stop()


if __name__ == "__main__":
    main()
