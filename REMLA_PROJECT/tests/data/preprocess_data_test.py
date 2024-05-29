import os
import sys
import pytest
from unittest import mock
import dvc.api
from remla_preprocess.pre_processing import MLPreprocessor

# Add the parent directory of `src` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocess_data import data_preprocessing

# Mock DVC params
mock_params = {
    "raw_data_path": "/path/to/raw_data/",
    "processed_data_path": "/path/to/processed_data/",
    "tokenizer_path": "/path/to/tokenizer/",
}

@pytest.fixture
def mock_dvc_params():
    with mock.patch.object(dvc.api, 'params_show', return_value=mock_params):
        yield

@pytest.fixture
def mock_mlpreprocessor():
    with mock.patch('remla_preprocess.pre_processing.MLPreprocessor', autospec=True) as mock_processor:
        instance = mock_processor.return_value
        instance.split_data_content.return_value = ["data"]
        instance.tokenize_pad_encode_data.return_value = {
            "train": ["encoded_train_data"],
            "test": ["encoded_test_data"],
            "val": ["encoded_val_data"],
            "tokenizer": "tokenizer_object",
            "char_index": "char_index_object"
        }
        yield instance

@pytest.fixture
def mock_os():
    with mock.patch('data_preprocessing.os.makedirs') as mock_makedirs, \
         mock.patch('data_preprocessing.os.path.exists', return_value=False):
        yield mock_makedirs

@pytest.fixture
def mock_save_pkl():
    with mock.patch.object(MLPreprocessor, 'save_pkl', autospec=True) as mock_save:
        yield mock_save

@pytest.fixture
def mock_load_txt():
    with mock.patch.object(MLPreprocessor, 'load_txt', return_value="dummy_data") as mock_load:
        yield mock_load

def test_data_preprocessing(mock_mlpreprocessor, mock_os, mock_save_pkl, mock_load_txt):
    data_preprocessing()

    # Check that load_txt was called with correct paths
    mock_load_txt.assert_any_call("/path/to/raw_data/train.txt")
    mock_load_txt.assert_any_call("/path/to/raw_data/test.txt")
    mock_load_txt.assert_any_call("/path/to/raw_data/val.txt")

    # Check that split_data_content was called
    mock_mlpreprocessor.split_data_content.assert_called()

    # Check that tokenize_pad_encode_data was called
    mock_mlpreprocessor.tokenize_pad_encode_data.assert_called_once_with(
        ["data"], ["data"], ["data"]
    )

    # Check that os.makedirs was called for processed_data_path and tokenizer_path
    mock_os.assert_any_call("/path/to/processed_data/")
    mock_os.assert_any_call("/path/to/tokenizer/")

    # Check that save_pkl was called for each key in trained_data
    assert mock_save_pkl.call_count == 5
    mock_save_pkl.assert_any_call(mock_mlpreprocessor, "encoded_train_data", "/path/to/processed_data/train.pkl")
    mock_save_pkl.assert_any_call(mock_mlpreprocessor, "encoded_test_data", "/path/to/processed_data/test.pkl")
    mock_save_pkl.assert_any_call(mock_mlpreprocessor, "encoded_val_data", "/path/to/processed_data/val.pkl")
    mock_save_pkl.assert_any_call(mock_mlpreprocessor, "tokenizer_object", "/path/to/tokenizer/tokenizer.pkl")
    mock_save_pkl.assert_any_call(mock_mlpreprocessor, "char_index_object", "/path/to/tokenizer/char_index.pkl")
