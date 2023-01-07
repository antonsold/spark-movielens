FROM apache/spark-py

USER root

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt