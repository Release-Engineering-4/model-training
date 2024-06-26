import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from REMLA_PROJECT.src.models.model1.train import train_model


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
    "hyperparameter,values",
    [
        ("batch_train", [16, 32, 64]),
        ("epoch", [10, 20, 30]),
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

        # Set up mock data
        mock_load_pkl.side_effect = [
            mock_data["x_train"],
            mock_data["y_train"],
            mock_data["x_val"],
            mock_data["y_val"],
        ]

        for value in values:
            # Set up mock parameters
            mock_params = {
                "model_path": "REMLA_PROJECT/models/",
                "processed_data_path": "REMLA_PROJECT/data/processed/",
                "trained_model_path": "REMLA_PROJECT/models/trained/",
                "batch_train": 32,
                "epoch": 10,
                "optimizer": "adam",
                "loss_function": "binary_crossentropy",
            }
            mock_params[hyperparameter] = value
            mock_params_show.return_value = mock_params

            # Run the training function with the mock model
            train_model(model=mock_model)

            # Check if the model was compiled and fit with the correct parameters
            mock_model.compile.assert_called_once_with(
                loss=mock_params["loss_function"],
                optimizer=mock_params["optimizer"],
                metrics=["accuracy"],
            )
            mock_model.fit.assert_called_once_with(
                mock_data["x_train"],
                mock_data["y_train"],
                batch_size=mock_params["batch_train"],
                epochs=mock_params["epoch"],
                shuffle=True,
                validation_data=(mock_data["x_val"], mock_data["y_val"]),
            )

            # Check if the model was saved
            mock_model.save.assert_called_once_with(
                mock_params["trained_model_path"] + "trained_model.h5"
            )

            # Reset mocks for the next iteration
            mock_model.compile.reset_mock()
            mock_model.fit.reset_mock()
            mock_model.save.reset_mock()
