import os
import sys

import yaml

from model import Model
from utils import Utils

CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")



class Predictor:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.spark = Utils.init_spark(**self.config["spark"])

    def predict(self):
        df = Utils.load_raw_data(self.spark, self.config["data_path"])
        model, success = Model.load_from_disk(
            self.config["models"]["tf_path"],
            self.config["models"]["idf_path"],
            self.config["models"]["normalizer_path"],
            self.config["models"]["brp_path"],
        )

        if not success:
            model.log.error("Error loading models. No predictions will be generated.")
            self.spark.stop()
            sys.exit(1)

        users_to_predict = Utils.get_users_to_predict(
            df, self.config["n_samples_to_predict"], self.config["random_seed"]
        )
        model.log.info("Generating predictions")
        preds = model.predict(df, users_to_predict)
        preds.show()
        self.spark.stop()


def main():
    predictor = Predictor(CONFIG_PATH)
    predictor.predict()


if __name__ == "__main__":
    main()
