import os
import sys
import pytest
from unittest import mock
from unittest.mock import MagicMock
import dvc.api
from remla_preprocess.pre_processing import MLPreprocessor

# Add the parent directory of `src` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.model1.train import train_model

# Mock DVC params
mock_params = {
    "model_path": "/path/to/model/",
    "processed_data_path": "/path/to/processed_data/",
    "trained_model_path": "/path/to/trained_model/",
    "loss_function": "binary_crossentropy",
    "optimizer": "adam",
    "batch_train": 32,
    "epoch": 10
}

@pytest.fixture
def mock_dvc_params():
    with mock.patch.object(dvc.api, 'params_show', return_value=mock_params):
        yield

@pytest.fixture
def mock_load_model():
    with mock.patch('keras.models.load_model') as mock_load_model:
        mock_model = MagicMock()
        mock_model.compile.return_value = None
        mock_model.fit.return_value = None
        mock_model.save.return_value = None
        mock_load_model.return_value = mock_model
        yield mock_load_model

@pytest.fixture
def mock_mlpreprocessor():
    with mock.patch.object(MLPreprocessor, 'load_pkl', side_effect=[[[1]], [1], [[1]], [1]]) as mock_load_pkl, \
         mock.patch.object(MLPreprocessor, 'save_pkl', autospec=True) as mock_save_pkl:
        yield mock_load_pkl, mock_save_pkl

@pytest.fixture
def mock_os():
    with mock.patch('os.makedirs') as mock_makedirs, \
         mock.patch('os.path.exists', return_value=False) as mock_exists:
        yield mock_makedirs, mock_exists

def test_train_model(mock_dvc_params, mock_load_model, mock_mlpreprocessor, mock_os):
    mock_load_pkl, mock_save_pkl = mock_mlpreprocessor
    mock_makedirs, mock_exists = mock_os

    train_model()

    # Check that load_model was called with the correct path
    mock_load_model.assert_called_once_with("/path/to/model/model.h5")

    # Check that load_pkl was called with the correct paths
    mock_load_pkl.assert_any_call("/path/to/processed_data/url_train.pkl")
    mock_load_pkl.assert_any_call("/path/to/processed_data/label_train.pkl")
    mock_load_pkl.assert_any_call("/path/to/processed_data/url_val.pkl")
    mock_load_pkl.assert_any_call("/path/to/processed_data/label_val.pkl")

    # Check that os.makedirs was called for trained_model_path
    mock_makedirs.assert_called_once_with("/path/to/trained_model/")

    # Check that the model save method was called with the correct path
    mock_model = mock_load_model.return_value
    mock_model.save.assert_called_once_with("/path/to/trained_model/trained_model.h5")

    # Ensure that model.compile was called with the correct arguments
    mock_model.compile.assert_called_once_with(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # Ensure that model.fit was called with the correct arguments
    mock_model.fit.assert_called_once_with(
        [[1]],
        [1],
        batch_size=32,
        epochs=10,
        shuffle=True,
        validation_data=([[1]], [1])
    )
