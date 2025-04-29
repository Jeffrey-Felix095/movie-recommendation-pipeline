import argparse
from typing import Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, explode
import logging
import mlflow
import mlflow.spark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "generate_recommendations") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")  # Ajusta según tu cluster
        .getOrCreate()
    )
    return spark


def load_best_model(
    spark: SparkSession, mlflow_run_id: str, artifact_path: str = "als_model"
):
    model_uri = f"runs:/{mlflow_run_id}/{artifact_path}"
    logger.info(f"Loading model from MLflow URI: {model_uri}")
    pipeline_model = mlflow.spark.load_model(model_uri)
    logger.info("Model loaded successfully.")

    # Extraer la etapa de ALS
    als_model = pipeline_model.stages[0]

    return als_model


def read_users_and_movies_data(
    spark: SparkSession, users_path: str, movies_path: str
) -> Tuple[DataFrame, DataFrame]:
    """Reads clean user and movie data (to get IDs and titles)."""
    logger.info(f"Reading users data from: {users_path}")
    users_df = spark.read.parquet(users_path).select("user_id")

    logger.info(f"Reading movies data from: {movies_path}")
    movies_df = spark.read.parquet(movies_path).select("movie_id", "title")

    return users_df, movies_df


def generate_and_transform_recommendations(
    model, users_df: DataFrame, movies_df: DataFrame, num_recommendations: int = 10
) -> DataFrame:
    """Generates recommendations, transforms them, and matches them with movie titles."""
    logger.info(
        f"Generating top {num_recommendations} recommendations for all users..."
    )

    user_recs = model.recommendForUserSubset(users_df, num_recommendations)

    user_recs_exploded = user_recs.withColumn(
        "recommendation", explode(col("recommendations"))
    )

    transformed_recs = user_recs_exploded.select(
        col("user_id"),
        col("recommendation.movie_id").alias("movie_id_recommended"),
        col("recommendation.rating").alias("predicted_rating"),
    )

    logger.info("Joining recommendations with movie titles...")
    final_recommendations_df = transformed_recs.join(
        movies_df, transformed_recs.movie_id_recommended == movies_df.movie_id, "left"
    ).select(
        "user_id",
        "movie_id_recommended",
        "title",
        "predicted_rating",
    )

    return final_recommendations_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate movie recommendations using a trained ALS model."
    )
    parser.add_argument(
        "--model-run-id",
        type=str,
        required=True,
        help="MLflow Run ID containing the best trained ALS model.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        required=True,
        help="MLflow tracking server URI (e.g., http://localhost:5000).",
    )
    parser.add_argument(
        "--users-path",
        type=str,
        required=True,
        help="Path to the cleaned users Parquet data.",
    )
    parser.add_argument(
        "--movies-path",
        type=str,
        required=True,
        help="Path to the cleaned movies Parquet data (for titles).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the final recommendations in Parquet format.",
    )
    parser.add_argument(
        "--num-recommendations",
        type=int,
        default=10,
        help="Number of top recommendations to generate per user.",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    logger.info(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")

    spark = create_spark_session()

    best_model = load_best_model(spark, args.model_run_id)

    users_df, movies_df = read_users_and_movies_data(
        spark, args.users_path, args.movies_path
    )

    final_recommendations_df = generate_and_transform_recommendations(
        best_model,
        users_df,
        movies_df,
        args.num_recommendations,
    )

    logger.info(f"Saving final recommendations to: {args.output_path}")
    final_recommendations_df.write.mode("overwrite").csv(args.output_path)

    logger.info("Recommendation generation and saving completed successfully.")

    spark.stop()


if __name__ == "__main__":
    main()
