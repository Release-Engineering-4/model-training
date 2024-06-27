import os
import random
import pandas as pd
import pytest
from remla_preprocess.pre_processing import MLPreprocessor
import dvc.api
import gdown

params = dvc.api.params_show()

RAW_DATA_PATH = "REMLA_PROJECT/data/raw/"

@pytest.fixture()
def create_prediction_directory():
    if not os.path.exists("REMLA_PROJECT/models/predictions/"):
        os.mkdir("REMLA_PROJECT/models/predictions/")

@pytest.fixture()
def create_tokenizer_directory():
    if not os.path.exists("REMLA_PROJECT/models/tokenizer/"):
        os.mkdir("REMLA_PROJECT/models/tokenizer/")

@pytest.fixture(scope="session", autouse=True)
def download_data():
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1MRWfSFhFTCluhst_O-D_Alqhh0aEQrJp",
            output=RAW_DATA_PATH,
        )
        # Make sure all files are in the same directory
        # no matter if it is run locally or through workflow
        if os.path.exists(RAW_DATA_PATH + "raw/"):
            for filename in os.listdir(RAW_DATA_PATH + "raw/"):
                os.rename(RAW_DATA_PATH + "raw/" + filename, RAW_DATA_PATH + filename)

@pytest.fixture
def processor():
    return MLPreprocessor()


@pytest.fixture
def example_data():

    train_data = "phishing\thttp://example1.com\nlegitimate\thttp://example2.com"
    test_data = "legitimate\thttp://example3.com\nphishing\thttp://example4.com"
    val_data = "phishing\thttp://example5.com\nphishing\thttp://example6.com"
    return train_data, test_data, val_data


def test_feature_distribution(processor):
    test_data = MLPreprocessor.load_txt(RAW_DATA_PATH + "test.txt")
    split_test_data = processor.split_data_content(test_data)

    unique_labels = split_test_data["label"].unique()

    assert len(unique_labels) == 2, "Expected exactly two unique labels"
    assert 'legitimate' in unique_labels, "Expected 'legitimate' label"
    assert 'phishing' in unique_labels, "Expected 'phishing' label"


def test_url_length_distribution(processor):
    test_data = MLPreprocessor.load_txt(RAW_DATA_PATH + "test.txt")
    split_test_data = processor.split_data_content(test_data)

    split_test_data["url_length"] = split_test_data["url"].apply(len)

    assert split_test_data["url_length"].mean() <= 80, "Url avg longer than 80 chars"
    assert split_test_data["url_length"].mean() >= 10, "Url avg longer than 10 chars"


def test_url_null(processor):
    test_data = MLPreprocessor.load_txt(RAW_DATA_PATH + "test.txt")
    split_test_data = processor.split_data_content(test_data)

    null_urls = split_test_data["url"].isnull().sum()

    assert null_urls == 0, "There are null urls" 


def test_label_null(processor):
    test_data = MLPreprocessor.load_txt(RAW_DATA_PATH + "test.txt")
    split_test_data = processor.split_data_content(test_data)

    null_labels = split_test_data["label"].isnull().sum()

    assert null_labels == 0, "There are null labels"


def test_feature_data_consistency(processor, example_data):
    train_data, test_data, val_data = example_data
    split_train_data = processor.split_data_content(train_data)
    split_test_data = processor.split_data_content(test_data)
    split_val_data = processor.split_data_content(val_data)

    assert len(split_train_data) > 0
    assert len(split_test_data) > 0
    assert len(split_val_data) > 0


def test_robustness(processor, example_data):
    train_data, test_data, val_data = example_data

    split_train_data = processor.split_data_content(train_data)
    split_test_data = processor.split_data_content(test_data)
    split_val_data = processor.split_data_content(val_data)

    trained_data = processor.tokenize_pad_encode_data(
        split_train_data, split_test_data, split_val_data
    )

    shuffled_train_data = split_train_data.sample(frac=1)

    shuffled_trained_data = processor.tokenize_pad_encode_data(
        shuffled_train_data, split_test_data, split_val_data
    )

    assert trained_data["url_train"][0] in shuffled_trained_data["url_train"]
    assert trained_data["url_train"][1] in shuffled_trained_data["url_train"]


def test_ml_infrastructure(processor, example_data, create_prediction_directory, create_tokenizer_directory):
    train_data, test_data, val_data = example_data
    split_train_data = processor.split_data_content(train_data)
    split_test_data = processor.split_data_content(test_data)
    split_val_data = processor.split_data_content(val_data)

    processor.tokenize_pad_encode_data(
        split_train_data, split_test_data, split_val_data
    )

    assert os.path.exists(params["processed_data_path"])
    assert os.path.exists(params["tokenizer_path"])


@pytest.mark.parametrize("data_slice", ["url_train", "url_test", "url_val"])
def test_data_slices(processor, example_data, data_slice):
    train_data, test_data, val_data = example_data
    split_train_data = processor.split_data_content(train_data)
    split_test_data = processor.split_data_content(test_data)
    split_val_data = processor.split_data_content(val_data)

    trained_data = processor.tokenize_pad_encode_data(
        split_train_data, split_test_data, split_val_data
    )

    assert data_slice in trained_data
    assert len(trained_data[data_slice]) > 0


@pytest.mark.parametrize("data_slice", ["label_train", "label_test", "label_val"])
def test_data_slices(processor, example_data, data_slice):
    train_data, test_data, val_data = example_data
    split_train_data = processor.split_data_content(train_data)
    split_test_data = processor.split_data_content(test_data)
    split_val_data = processor.split_data_content(val_data)

    trained_data = processor.tokenize_pad_encode_data(
        split_train_data, split_test_data, split_val_data
    )

    assert data_slice in trained_data
    assert len(trained_data[data_slice]) > 0
