import importlib.util
import os
import sys
import numpy as np
import pytest
from remla_preprocess.pre_processing import MLPreprocessor
import dvc.api
import gdown
from keras.models import Sequential, load_model
from keras.layers import Dense
import time
import psutil

def load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


module_path = r"REMLA_PROJECT/src/models/model1/train.py"
module_name = "train_module"
train_module = load_module(module_path, module_name)
train_model = train_module.train_model


def load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


module_path = r"REMLA_PROJECT/src/models/model1/predict.py"
module_name = "predict_module"
predict_module = load_module(module_path, module_name)
predict_model = predict_module.predict


params = dvc.api.params_show()

TRAINED_MODEL_PATH = "REMLA_PROJECT/models/trained_model/"

RAW_DATA_PATH = "REMLA_PROJECT/data/raw/"


@pytest.fixture()
def create_predictions_directory():
    if not os.path.exists("REMLA_PROJECT/models/predictions/"):
        os.mkdir("REMLA_PROJECT/models/predictions/")


@pytest.fixture()
def create_trained_directory():
    if not os.path.exists("REMLA_PROJECT/models/trained_model/"):
        os.mkdir("REMLA_PROJECT/models/trained_model/")


@pytest.fixture(scope="session", autouse=True)
def download_data():
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1MRWfSFhFTCluhst_O-D_Alqhh0aEQrJp",
            output=RAW_DATA_PATH,
        )
    if os.path.exists(RAW_DATA_PATH + "raw/"):
        for filename in os.listdir(RAW_DATA_PATH + "raw/"):
            os.rename(RAW_DATA_PATH + "raw/" + filename, RAW_DATA_PATH + filename)


@pytest.fixture(scope="session", autouse=True)
def download_model():
    if not os.path.exists(TRAINED_MODEL_PATH):
        os.makedirs(TRAINED_MODEL_PATH)
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1eH-IwLO7k1M0VdhsV7lc9LFJEHsiWprW",
            output=TRAINED_MODEL_PATH,
        )
    if os.path.exists(TRAINED_MODEL_PATH + "rml_model/"):
        for filename in os.listdir(TRAINED_MODEL_PATH + "rml_model/"):
            os.rename(
                TRAINED_MODEL_PATH + "rml_model/" + filename, TRAINED_MODEL_PATH + filename
            )

@pytest.fixture
def simple_model():
    model = Sequential()
    model.add(Dense(10, input_shape=(200,), activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


@pytest.fixture
def processor():
    return MLPreprocessor()


@pytest.fixture
def single_params():
    return {
        "batch_train": 32,
        "epoch": 1,
        "optimizer": "sgd",
        "loss_function": "mse",
        "metrics": "accuracy",
    }


def test_pipeline(
    processor,
    simple_model,
    single_params,
    create_predictions_directory,
    create_trained_directory,
):
    train_data = processor.load_txt(RAW_DATA_PATH + "train.txt")
    val_data = processor.load_txt(RAW_DATA_PATH + "val.txt")
    test_data = processor.load_txt(RAW_DATA_PATH + "test.txt")
    good_model = load_model(TRAINED_MODEL_PATH + "own_trained_model.h5")
    split_train_data = processor.split_data_content(train_data)
    split_val_data = processor.split_data_content(val_data)
    split_test_data = processor.split_data_content(test_data)
    preprocessed_data = processor.tokenize_pad_encode_data(
        split_train_data[:500], split_val_data[:500], split_test_data[:500]
    )
    # We train the new model on slices of data only for time reasons (workflow is faster)
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    train_model(
        simple_model,
        preprocessed_data["url_train"],
        preprocessed_data["label_train"],
        preprocessed_data["url_val"],
        preprocessed_data["label_val"],
        single_params,
    )

    duration = time.time() - start_time
    memory_usage = process.memory_info().rss - initial_memory

    trained_simple_model = load_model(params["trained_model_path"] + "trained_model.h5")

    predict_model(
        trained_simple_model,
        preprocessed_data["url_test"][:10],
        preprocessed_data["label_test"][:10],
    )
    bad_predictions = processor.load_pkl(
        params["predictions_path"] + "label_pred_binary.pkl"
    )

    start_time2 = time.time()
    predict_model(good_model, preprocessed_data["url_test"][:10], preprocessed_data["label_test"][:10])
    duration2 = time.time() - start_time2
    good_predictions = processor.load_pkl(
        params["predictions_path"] + "label_pred_binary.pkl"
    )

    good_predictions = good_predictions.flatten()
    bad_predictions = bad_predictions.flatten()

    accuracy_model_1 = np.mean(good_predictions == preprocessed_data["label_test"][:10])
    accuracy_model_2 = np.mean(bad_predictions == preprocessed_data["label_test"][:10])

    assert duration < 150, "Test took too long"
    assert duration2 < 5, "Predictions took too long"
    assert memory_usage < 150 * 1024 * 1024, "Memory usage too high"
    assert accuracy_model_1 >= accuracy_model_2
