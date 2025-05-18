import os
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

from datetime import timedelta

import pendulum

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["test@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
}
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_tracking:5000")
AIRFLOW_HOME = "/opt/airflow"
SCRIPTS_PATH = f"{AIRFLOW_HOME}/scripts"
DATA_RAW_PATH = f"{AIRFLOW_HOME}/data/raw"
DATA_PROCESSED_PATH = f"{AIRFLOW_HOME}/data/processed"

with DAG(
    dag_id="movie_recommendation_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["pyspark", "mlflow", "recommendation", "etl"],
    doc_md="""
    ### Movie Recommendation Pipeline DAG
    This DAG orchestrates the process of ingesting, cleaning, transforming,
    training an ALS recommendation model, and generating recommendations
    using PySpark, Airflow, and MLflow.
    """,
) as dag:

    ingest_clean_task = BashOperator(
        task_id="ingest_parse_clean_raw_data",
        bash_command=(
            f"spark-submit {SCRIPTS_PATH}/ingest_parse_clean.py "
            f"--raw-path {DATA_RAW_PATH} "
            f"--output-path {DATA_PROCESSED_PATH}/cleaned"
        ),
    )

    transform_join_task = BashOperator(
        task_id="trasnform_join_data",
        bash_command=(
            f"spark-submit {SCRIPTS_PATH}/transform_join.py "
            f"--clean-data-path {DATA_PROCESSED_PATH}/cleaned "
            f"--output-path {DATA_PROCESSED_PATH}/joined"
        ),
    )

    calculate_avg_ratings_task = BashOperator(
        task_id="calculate_avg_ratings",
        bash_command=(
            f"spark-submit {SCRIPTS_PATH}/calculate_avg_ratings.py "
            f"--input-path {DATA_PROCESSED_PATH}/joined "
            f"--output-path {DATA_PROCESSED_PATH}/avg_ratings"
        ),
    )

    calculate_genre_popularity_task = BashOperator(
        task_id="calculate_genre_popularity",
        bash_command=(
            f"spark-submit {SCRIPTS_PATH}/calculate_genre_popularity.py "
            f"--input-path {DATA_PROCESSED_PATH}/joined "
            f"--output-path {DATA_PROCESSED_PATH}/genre_popularity"
        ),
    )

    prepare_als_data_task = BashOperator(
        task_id="prepare_als_data",
        bash_command=(
            f"spark-submit {SCRIPTS_PATH}/preparate_data_for_als.py "
            f"--input-ratings-path {DATA_PROCESSED_PATH}/cleaned/ratings.parquet "
            f"--output-train-path {DATA_PROCESSED_PATH}/als_training/train.parquet "
            f"--output-validation-path {DATA_PROCESSED_PATH}/als_training/validation.parquet"
        ),
    )

    train_als_task = BashOperator(
        task_id="train_als_model",
        bash_command=(
            f"spark-submit {SCRIPTS_PATH}/train_als_model.py "
            f"--input-train-path {DATA_PROCESSED_PATH}/als_training/train.parquet "
            f"--input-validation-path {DATA_PROCESSED_PATH}/als_training/validation.parquet "
            f"--mlflow-tracking-uri {MLFLOW_TRACKING_URI} "
            # El script train_als_model.py debe imprimir la run_id del mejor modelo en stdout
            # para que BashOperator la capture en XComs.
            f"|| exit 1"  # Asegura que la tarea falla si spark-submit falla
        ),
    )

    generate_recs_task = BashOperator(
        task_id="generate_recommendations",
        bash_command=(
            # Leer el run_id del archivo y guardarlo en una variable de entorno
            "RUN_ID=$(cat /opt/airflow/data/intermediate/best_run_id.txt) && "
            # Luego lanzar spark-submit usando esa variable
            f"spark-submit {SCRIPTS_PATH}/generate_recommendations.py "
            f"--mlflow-tracking-uri {MLFLOW_TRACKING_URI} "
            "--model-run-id $RUN_ID "
            f"--users-path {DATA_PROCESSED_PATH}/cleaned/users.parquet "
            f"--movies-path {DATA_PROCESSED_PATH}/cleaned/movies.parquet "
            f"--output-path {DATA_PROCESSED_PATH}/final/user_recommendations.parquet "
            "--num-recommendations 10"
        ),
    )

    ingest_clean_task >> transform_join_task

    transform_join_task >> [
        calculate_avg_ratings_task,
        calculate_genre_popularity_task,
        prepare_als_data_task,
    ]

    prepare_als_data_task >> train_als_task

    train_als_task >> generate_recs_task
