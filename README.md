# Nombre del Proyecto: Pipeline de Recomendación de Películas con Airflow, Spark y MLflow

## Descripción del Proyecto

Este proyecto implementa un pipeline de datos automatizado para construir un sistema de recomendación de películas utilizando el conjunto de datos **MovieLens**. El pipeline cubre desde la ingesta y procesamiento de datos hasta el entrenamiento de un modelo de recomendación ALS (Alternating Least Squares) con PySpark MLlib y la generación de recomendaciones.

La orquestación del pipeline se realiza con **Apache Airflow**, el seguimiento de experimentos y modelos se gestiona con **MLflow**, y todo el entorno se empaqueta para su fácil despliegue y reproducibilidad usando **Docker Compose**.

---

## Datasets utilizados

Este proyecto utiliza el dataset **MovieLens 1M**, que contiene:

- 1,000,209 calificaciones anónimas de ~3,900 películas
- Hechas por 6,040 usuarios
- Cada usuario calificó al menos 20 películas
- Las calificaciones están en escala de 1 a 5 estrellas

**Fuente oficial**: [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)

**Licencia de uso**:

- El dataset puede usarse solo con fines de investigación
- No puede redistribuirse sin permiso
- Se debe citar la siguiente publicación al usarlo:

> F. Maxwell Harper and Joseph A. Konstan. 2015. _The MovieLens Datasets: History and Context._ ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19. DOI: [10.1145/2827872](http://dx.doi.org/10.1145/2827872)

---

## Tecnologías Utilizadas

- **Apache Airflow:** Orquestación de flujos de trabajo (DAGs).
- **Apache Spark (PySpark):** Procesamiento de datos distribuido y entrenamiento del modelo ALS.
- **MLflow:** Seguimiento de experimentos (MLflow Tracking Server), registro de parámetros, métricas y modelos.
- **PostgreSQL:** Base de datos de metadatos de Airflow.
- **Docker & Docker Compose:** Contenerización y gestión del entorno de servicios.
- **Python 3.11:** Lenguaje de programación principal.

## Configuración Inicial (`.env`)

Copia el archivo `.env.example` (si tienes uno, o crea uno llamado `.env`) en la raíz del proyecto. Edítalo para configurar las variables necesarias:

- `AIRFLOW_UID`, `AIRFLOW_GID`: Tu ID de usuario y grupo local para asegurar permisos correctos en los volúmenes montados.
- `_AIRFLOW_WWW_USER_USERNAME`, `_AIRFLOW_WWW_USER_PASSWORD`: Credenciales para el usuario administrador de la UI de Airflow.
- `AIRFLOW_VERSION`, `PYTHON_VERSION`: Versiones de Airflow y Python a usar (deben coincidir con el `Dockerfile`).
- `MLFLOW_TRACKING_URI`: Debe ser `http://mlflow_tracking:5000` para la comunicación entre contenedores.
- `AIRFLOW__WEBSERVER__SECRET_KEY`: Una clave secreta aleatoria para la seguridad de Airflow. Puedes generarla con `python -c 'import secrets; print(secrets.token_hex(16))'`.

## Puesta en Marcha del Entorno (Docker Compose)

Sigue estos pasos para levantar todo el entorno. Es importante seguir el orden para asegurar una inicialización correcta de la base de datos.

1.  **Clonar el repositorio** (si aún no lo has hecho).
2.  **Navegar al directorio del proyecto** en tu terminal.
3.  **Asegurarse de tener los datos de MovieLens** en `./data/raw/`. Si no los tienes, descárgalos y colócalos ahí.
4.  **Limpiar cualquier entorno Docker anterior** y eliminar los volúmenes (especialmente la DB de Airflow para empezar limpio):
    ```bash
    docker compose down -v
    ```
5.  **Ejecutar la inicialización de Airflow** como un paso separado. Esto construye la imagen Docker y prepara la base de datos.
    ```bash
    docker compose up airflow-init --build --force-recreate
    ```
6.  **Levantar los servicios principales** (Webserver, Scheduler, MLflow, Postgres) en modo detached:
    ```bash
    docker compose up -d --force-recreate airflow-webserver airflow-scheduler mlflow_tracking postgres
    ```

## Acceso a las Interfaces

- **Airflow UI:** Abre tu navegador y ve a `http://localhost:8080`. Inicia sesión con las credenciales configuradas en tu `.env`.
- **MLflow UI:** Abre tu navegador y ve a `http://localhost:5000`.

## Pasos del Pipeline

El DAG `movie_recommendation_pipeline` ejecuta las siguientes tareas secuencialmente:

1.  `ingest_parse_clean_raw_data`: Lee datos crudos de `.dat`, limpia y guarda en `./data/processed/cleaned`.
2.  `trasnform_join_data`: Combina los datos limpios y guarda en `./data/processed/joined`.
3.  `calculate_avg_ratings`: Calcula ratings promedio y guarda resultados.
4.  `calculate_genre_popularity`: Calcula popularidad por género y guarda resultados.
5.  `prepare_als_data`: Prepara datos para el entrenamiento ALS y guarda en `./data/processed/als_training`.
6.  `train_als_model`: Entrena el modelo ALS usando PySpark MLlib, registra parámetros, métricas y el modelo en MLflow.
7.  `generate_recommendations`: Carga el modelo de MLflow, genera recomendaciones y guarda el resultado final en `./data/processed/final/`.

## Salidas del Pipeline

El resultado final del pipeline es el archivo de recomendaciones para usuarios, guardado en `./data/processed/final/user_recommendations.csv`. También se generan archivos intermedios en otras subcarpetas de `./data/processed/`.
