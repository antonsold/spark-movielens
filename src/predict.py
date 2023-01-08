import utils
import sys
import os
from model import Model
import yaml

CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    spark = utils.init_spark(**config["spark"])
    df = utils.load_raw_data(spark, config["data_path"])
    model, success = Model.load_from_disk(
        config["models"]["tf_path"],
        config["models"]["idf_path"],
        config["models"]["normalizer_path"],
        config["models"]["brp_path"]
    )

    if not success:
        model.log.error("Error loading models. No predictions will be generated.")
        spark.stop()
        sys.exit(1)

    users_to_predict = utils.get_users_to_predict(
        df,
        config["n_samples_to_predict"],
        config["random_seed"]
    )
    model.log.info("Generating predictions")
    preds = model.predict(df, users_to_predict)
    preds.show()
    spark.stop()


if __name__ == "__main__":
    main()
