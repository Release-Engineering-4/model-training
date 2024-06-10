import os
import sys
import pytest
from unittest import mock
from unittest.mock import MagicMock
import dvc.api
from keras.models import Sequential
from keras.layers import (Embedding,
                          Dense)
from remla_preprocess.pre_processing import MLPreprocessor

# Add the parent directory of `src` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.model1.model import model_definition

# Mock DVC params
mock_params = {
    "tokenizer_path": "/path/to/tokenizer/",
    "max_input_length": 100,
    "categories": ["cat1", "cat2", "cat3"],
    "model_path": "/path/to/model/"
}

@pytest.fixture
def mock_dvc_params():
    with mock.patch.object(dvc.api, 'params_show', return_value=mock_params):
        yield

@pytest.fixture
def mock_mlpreprocessor():
    with mock.patch.object(MLPreprocessor, 'load_pkl', return_value={"a": 1, "b": 2}) as mock_load_pkl:
        yield mock_load_pkl

@pytest.fixture
def mock_os():
    with mock.patch('os.makedirs') as mock_makedirs, \
         mock.patch('os.path.exists', return_value=False):
        yield mock_makedirs

@pytest.fixture
def mock_model_save():
    with mock.patch.object(Sequential, 'save', autospec=True) as mock_save:
        yield mock_save

def test_model_definition(mock_mlpreprocessor, mock_os, mock_model_save):
    model_definition()

    # Check that load_pkl was called with the correct path
    mock_mlpreprocessor.assert_called_once_with("REMLA_PROJECT/models/tokenizer/char_index.pkl")

    # Check that os.makedirs was called for model_path
    mock_os.assert_called_once_with("REMLA_PROJECT/models/model/")

    # Check that the model save method was called with the correct path
    mock_model_save.assert_called_once_with(mock.ANY, "REMLA_PROJECT/models/model/model.h5")

    # Ensure the correct model architecture is created
    mock_model = mock_model_save.call_args[0][0]
    assert isinstance(mock_model, Sequential)
    assert len(mock_model.layers) == 21

    # Check the configuration of the first layer (Embedding layer)
    embedding_layer = mock_model.layers[0]
    assert isinstance(embedding_layer, Embedding)
    assert embedding_layer.input_dim == 3  # voc_size + 1 where voc_size = 2
    assert embedding_layer.output_dim == 50
    assert embedding_layer.input_length == 100

    # Check the configuration of the last layer (Dense layer)
    dense_layer = mock_model.layers[-1]
    assert isinstance(dense_layer, Dense)
    assert dense_layer.units == 2  # len(params["categories"]) - 1
    assert dense_layer.activation.__name__ == "sigmoid"
