import os
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd

# This script loads the data from huggingface in case we are not using a cloud provider
# This is more neat
login(os.environ.get("HUGGINGFACE_TOKEN"))

HUGGINGFACE_REPOSITORY = "Razvan27/remla_phishing_url"


def load_data():
    """
    Load data from huggingface
    """
    training_data = load_dataset(HUGGINGFACE_REPOSITORY, split="train")
    testing_data = load_dataset(HUGGINGFACE_REPOSITORY, split="test")
    validation_data = load_dataset(HUGGINGFACE_REPOSITORY, split="validation")
    df_train, df_test, df_val = (
        pd.DataFrame(training_data),
        pd.DataFrame(testing_data),
        pd.DataFrame(validation_data),
    )
    df_train.to_csv("REMLA_PROJECT/data/raw/train_data.csv", index=False)
    df_test.to_csv("REMLA_PROJECT/data/raw/test_data.csv", index=False)
    df_val.to_csv("REMLA_PROJECT/data/raw/validation_data.csv", index=False)


if __name__ == "__main__":
    load_data()
