from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F


def init_spark(master: str, app_name: str) -> SparkSession:
    return SparkSession.builder.master(master).appName(app_name).getOrCreate()


def load_raw_data(spark: SparkSession, path: str) -> DataFrame:
    df_schema = StructType(
        [
            StructField("user_id", IntegerType(), True),
            StructField("film_id", IntegerType(), True),
        ]
    )
    return (
        spark.read.csv(path, schema=df_schema)
        .groupBy("user_id")
        .agg(F.collect_list("film_id").alias("film_ids"))
    )


def get_users_to_predict(data: DataFrame, n: int, random_seed: int):
    fraction = n / data.count()
    return data.sample(fraction, random_seed).select("user_id")
