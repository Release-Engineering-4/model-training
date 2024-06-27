import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import importlib.util


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


@pytest.fixture
def mock_data():
    return {
        "x_train": np.random.rand(100, 10),
        "y_train": np.random.randint(0, 2, 100),
        "x_val": np.random.rand(20, 10),
        "y_val": np.random.randint(0, 2, 20),
    }


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.compile = MagicMock()
    model.fit = MagicMock()
    model.save = MagicMock()
    return model


@pytest.mark.parametrize(
    "hyperparameter, values",
    [
        ("batch_train", [16, 32, 64]),
        ("epoch", [1, 2, 3]),
        ("optimizer", ["adam", "sgd", "rmsprop"]),
        ("loss_function", ["binary_crossentropy", "mse"]),
    ],
)
def test_hyperparameter_impact(mock_data, mock_model, hyperparameter, values):
    with patch("dvc.api.params_show") as mock_params_show, patch(
        "remla_preprocess.pre_processing.MLPreprocessor.load_pkl"
    ) as mock_load_pkl, patch("os.path.exists", return_value=True), patch(
        "os.makedirs"
    ):

        def side_effect(arg):
            if "url_train.pkl" in arg:
                return mock_data["x_train"]
            elif "label_train.pkl" in arg:
                return mock_data["y_train"]
            elif "url_val.pkl" in arg:
                return mock_data["x_val"]
            elif "label_val.pkl" in arg:
                return mock_data["y_val"]
            else:
                raise ValueError(f"Unexpected argument: {arg}")

        mock_load_pkl.side_effect = side_effect

        for value in values:
            mock_params = {
                "model_path": "REMLA_PROJECT/models/",
                "processed_data_path": "REMLA_PROJECT/data/processed/",
                "trained_model_path": "REMLA_PROJECT/models/trained_model/",
                "batch_train": 32,
                "epoch": 1,
                "optimizer": "adam",
                "loss_function": "binary_crossentropy",
            }
            mock_params[hyperparameter] = value
            mock_params_show.return_value = mock_params

            custom_params = {
                "batch_train": mock_params["batch_train"],
                "epoch": mock_params["epoch"],
                "optimizer": mock_params["optimizer"],
                "loss_function": mock_params["loss_function"],
                "metrics": [
                    "accuracy"
                ], 
            }
                   
            train_model(
                model=mock_model,
                x_train=mock_data["x_train"],
                y_train=mock_data["y_train"],
                x_val=mock_data["x_val"],
                y_val=mock_data["y_val"],
                custom_params=custom_params,
            )

            mock_model.compile.assert_called_once_with(
                loss=custom_params["loss_function"],
                optimizer=custom_params["optimizer"],
                metrics=custom_params["metrics"],
            )

            mock_model.fit.assert_called_once_with(
                mock_data["x_train"],
                mock_data["y_train"],
                batch_size=custom_params["batch_train"],
                epochs=custom_params["epoch"],
                shuffle=True,
                validation_data=(mock_data["x_val"], mock_data["y_val"]),
            )

            mock_model.save.assert_called_once_with(
                mock_params["trained_model_path"] + "trained_model.h5"
            )

            mock_model.compile.reset_mock()
            mock_model.fit.reset_mock()
            mock_model.save.reset_mock()
