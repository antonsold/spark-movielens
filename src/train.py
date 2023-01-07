import os
import yaml

import utils
from model import Model

CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    spark = utils.init_spark(**config["spark"])
    df = utils.load_raw_data(spark, config["data_path"])
    model = Model(
        config["models"]["tf_path"],
        config["models"]["idf_path"],
        config["models"]["normalizer_path"],
        config["models"]["brp_path"],
    )
    model.train(df, config["models"]["num_features"], "film_ids")
    model.save_models()
    spark.stop()


if __name__ == "__main__":
    main()
