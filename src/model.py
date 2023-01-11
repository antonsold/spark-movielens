import traceback
from typing import List, Optional

from pyspark.ml.feature import (IDF, BucketedRandomProjectionLSH,
                                BucketedRandomProjectionLSHModel, HashingTF,
                                IDFModel, Normalizer)
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame

from logger import Logger

SHOW_LOG = True


class Model:
    def __init__(
        self, tf_path: str, idf_path: str, normalizer_path: str, brp_path: str
    ) -> None:
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        self.tf_path = tf_path
        self.idf_path = idf_path
        self.normalizer_path = normalizer_path
        self.brp_path = brp_path
        self.tf_model: Optional[HashingTF] = None
        self.idf_model: Optional[IDF] = None
        self.l2_normalizer: Optional[Normalizer] = None
        self.brp: Optional[BucketedRandomProjectionLSHModel] = None
        return

    def load_models(self) -> bool:
        try:
            self.tf_model = HashingTF.load(self.tf_path)
            self.idf_model = IDFModel.load(self.idf_path)
            self.l2_normalizer = Normalizer.load(self.normalizer_path)
            self.brp = BucketedRandomProjectionLSHModel.load(self.brp_path)
        except:
            self.log.error(traceback.format_exc())
            return False
        return True

    def save_models(self) -> bool:
        try:
            for model, path in zip(
                [self.tf_model, self.idf_model, self.l2_normalizer, self.brp],
                [self.tf_path, self.idf_path, self.normalizer_path, self.brp_path],
            ):
                model.write().overwrite().save(path)
        except:
            self.log.error(traceback.format_exc())
            return False
        return True

    def train(self, raw_data: DataFrame, num_features: int, input_col: str) -> None:
        # TF
        self.log.info("Training TF")
        self.tf_model = HashingTF(
            numFeatures=num_features, inputCol=input_col, outputCol="tf_features"
        )
        df_tf = self.tf_model.transform(raw_data)

        # IDF
        self.log.info("Training IDF")
        self.idf_model = IDF(inputCol="tf_features", outputCol="tfidf_features").fit(
            df_tf
        )
        df_tfidf = self.idf_model.transform(df_tf)

        # ANN for cosine similarities is not implemented in Spark, hence we normalize vectors and use Euclidean distance
        self.log.info("Normalizing and training ANN model")
        self.l2_normalizer = Normalizer(
            p=2, inputCol="tfidf_features", outputCol="norm_features"
        )
        df_norm = self.l2_normalizer.transform(df_tfidf)
        self.brp = BucketedRandomProjectionLSH(
            inputCol="norm_features",
            outputCol="hashes",
            bucketLength=0.5,
            numHashTables=5,
        ).fit(df_norm)
        return

    def predict(self, raw_data: DataFrame, user_ids: DataFrame,) -> DataFrame:
        self.log.info("Applying TF")
        data = self.tf_model.transform(raw_data)

        self.log.info("Applying IDF")
        data = self.idf_model.transform(data)

        self.log.info("Normalizing")
        data = self.l2_normalizer.transform(data)

        self.log.info("Using ANN and filtering")
        data = self.brp.transform(data)
        df_to_predict = user_ids.join(data, on="user_id")

        # Calculating pairwise distances between feature vectors of users to predict and all other users
        similarities = (
            self.brp.approxSimilarityJoin(
                df_to_predict, data, threshold=1.5, distCol="EuclideanDistance"
            )
            .where("datasetA.user_id != datasetB.user_id")
            .select("datasetA.user_id", "datasetB.film_ids", "EuclideanDistance")
        )

        # Selecting users with minimal distance
        window = Window.partitionBy("user_id")
        predictions = (
            similarities.withColumn(
                "minDistance", F.min("EuclideanDistance").over(window)
            )
            .where("minDistance == EuclideanDistance")
            .drop("minDistance")
        )

        return predictions

    @classmethod
    def load_from_disk(
        cls, tf_path: str, idf_path: str, normalizer_path: str, brp_path: str
    ):
        model = cls(tf_path, idf_path, normalizer_path, brp_path)
        success = model.load_models()
        return model, success
