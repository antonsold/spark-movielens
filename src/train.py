import os
import yaml

from utils import Utils
from model import Model

CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")


class Trainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.spark = Utils.init_spark(**self.config["spark"])

    def train(self):
        df = Utils.load_raw_data(self.spark, self.config["data_path"])
        model = Model(
            self.config["models"]["tf_path"],
            self.config["models"]["idf_path"],
            self.config["models"]["normalizer_path"],
            self.config["models"]["brp_path"],
        )
        model.train(df, self.config["models"]["num_features"], "film_ids")
        model.save_models()
        self.spark.stop()


def main():
    trainer = Trainer(CONFIG_PATH)
    trainer.train()


if __name__ == "__main__":
    main()
