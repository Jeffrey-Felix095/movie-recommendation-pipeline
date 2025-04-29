FROM apache/airflow:2.9.0-python3.11

USER root

RUN apt update && apt install -y --no-install-recommends \
    passwd procps openjdk-17-jre \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"

USER airflow

WORKDIR /opt/airflow

COPY requirements.txt /opt/airflow/

RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

