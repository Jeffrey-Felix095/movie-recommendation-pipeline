# scripts/train_als_model.py

import argparse
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
import pandas as pd


def create_spark_session(app_name: str = "train_als_model") -> SparkSession:
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark


def read_als_data(
    spark: SparkSession, train_path: str, validation_path: str
) -> tuple[DataFrame, DataFrame]:
    print(f"Reading training data from: {train_path}")
    training_data = spark.read.parquet(train_path)

    print(f"Reading validation data from: {validation_path}")
    validation_data = spark.read.parquet(validation_path)

    return training_data, validation_data


def define_param_grid():
    param_grid = [
        {"rank": 10, "regParam": 0.01},
        {"rank": 10, "regParam": 0.1},
        {"rank": 15, "regParam": 0.01},
        {"rank": 15, "regParam": 0.1},
    ]
    return param_grid


def train_and_tune_als(
    spark: SparkSession,
    training_data: DataFrame,
    validation_data: DataFrame,
    param_grid: list,
) -> str:
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )

    best_rmse = float("inf")
    best_run_id = None
    best_params = None

    
    als = ALS(
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        coldStartStrategy="drop",
    )

    
    with mlflow.start_run(run_name="ALS Hyperparameter Tuning"):
        print("Starting ALS hyperparameter tuning with MLflow...")
        mlflow_parent_run_id = (
            mlflow.active_run().info.run_id
        )

        for i, params in enumerate(param_grid):
            with mlflow.start_run(
                run_name=f"Run {i+1} (rank={params['rank']}, regParam={params['regParam']})",
                nested=True,
            ) as nested_run:
                print(f"  > Training model with parameters: {params}")
                current_run_id = (
                    nested_run.info.run_id
                )

                mlflow.log_params(params)

                als.setParams(**params)

                try:
                    model = als.fit(training_data)
                    print("  > Training completed.")

                    predictions = model.transform(validation_data)

                    rmse = evaluator.evaluate(predictions)
                    print(f"  > RMSE for parameters {params}: {rmse}")

                    mlflow.log_metric("rmse", rmse)

                    example_input_pd = pd.DataFrame({"userID": [1], "movieID": [10]})

                    mlflow.spark.log_model(
                        artifact_path="als_model",
                        spark_model=model,
                        input_example=example_input_pd,
                    )

                    print(f"  > Model logged in nested run {current_run_id}")

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_run_id = (
                            current_run_id 
                        )
                        best_params = params
                        print(
                            f"  > New best RMSE found: {best_rmse} in run {best_run_id}"
                        )

                except Exception as e:
                    print(
                        f"  > Error during training or evaluation for parameters {params}: {e}"
                    )
                    mlflow.log_param("error", str(e))

        print("\nHyperparameter tuning process completed.")
        print(f"Best overall RMSE found: {best_rmse}")
        print(f"Best parameters: {best_params}")
        print(f"Run ID with the best result: {best_run_id}")

        mlflow.log_metric("best_overall_rmse", best_rmse)
        mlflow.log_param("best_run_id", best_run_id)
        mlflow.log_param(
            "best_params_used", str(best_params)
        )

    print(f"\nMLflow experiment finished. Best model is in run: {best_run_id}")
    return best_run_id


def main():
    parser = argparse.ArgumentParser(
        description="Train and tune ALS model with MLflow."
    )
    parser.add_argument(
        "--input-train-path",
        type=str,
        required=True,
        help="Path to the training data Parquet file.",
    )
    parser.add_argument(
        "--input-validation-path",
        type=str,
        required=True,
        help="Path to the validation data Parquet file.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        required=True,
        help="MLflow tracking server URI (e.g., http://localhost:5000).",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    print(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")

    spark = create_spark_session()

    training_data, validation_data = read_als_data(
        spark, args.input_train_path, args.input_validation_path
    )

    param_grid = define_param_grid()

    best_run_id = train_and_tune_als(spark, training_data, validation_data, param_grid)

    output_dir = "/opt/airflow/data/intermediate"

    output_file = os.path.join(output_dir, "best_run_id.txt")

    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(best_run_id)

    spark.stop()


if __name__ == "__main__":
    main()
