import os
import random
import re
import numpy as np
import pandas as pd
import pytest
from remla_preprocess.pre_processing import MLPreprocessor
import gdown


RAW_DATA_PATH = "REMLA_PROJECT/data/raw/"


@pytest.fixture()
def download_data(scope="session", autouse=True):
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1MRWfSFhFTCluhst_O-D_Alqhh0aEQrJp",
            output=RAW_DATA_PATH,
        )


@pytest.fixture
def processor():
    return MLPreprocessor()


def test_url_length(processor):
    train_data = MLPreprocessor.load_txt(RAW_DATA_PATH + "test.txt")
    split_test_data = processor.split_data_content(train_data[:500])

    assert all(
        len(url) > 0 for url in split_test_data["url"]
    ), "Some URLs have zero length."


def test_label_distribution(processor):
    train_data = MLPreprocessor.load_txt(RAW_DATA_PATH + "test.txt")
    split_test_data = processor.split_data_content(train_data[:500])

    split_test_data["binary_labels"] = split_test_data["label"].apply(
        lambda x: 0 if x == "legitimate" else 1
    )

    class_counts = np.bincount(split_test_data["binary_labels"])
    assert (
        np.abs(class_counts[1] - class_counts[0]) <= 1
    ), "Class distribution is not balanced."


def test_label_validity(processor):
    train_data = MLPreprocessor.load_txt(RAW_DATA_PATH + "test.txt")
    split_test_data = processor.split_data_content(train_data[:500])

    assert all(
        label in ["legitimate", "phishing"] for label in split_test_data["label"]
    ), "Labels are not legitimate or phishing."


def test_url_format(processor):
    train_data = MLPreprocessor.load_txt(RAW_DATA_PATH + "test.txt")
    split_test_data = processor.split_data_content(train_data[:500])
    assert all(validate_url_format(url) for url in split_test_data["url"]), "Some URLs are not in a valid format."


def validate_url_format(url):
    regex_pattern = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    return re.match(regex_pattern, url) is not None
