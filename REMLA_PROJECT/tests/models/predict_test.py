import pytest
from unittest import mock
from unittest.mock import MagicMock
import dvc.api
import numpy as np
from remla_preprocess.pre_processing import MLPreprocessor
from src.models.model1.predict import predict

# Mock DVC params
mock_params = {
    "trained_model_path": "/path/to/trained_model/",
    "processed_data_path": "/path/to/processed_data/",
    "predictions_path": "/path/to/predictions/"
}

@pytest.fixture
def mock_dvc_params():
    with mock.patch.object(dvc.api, 'params_show', return_value=mock_params):
        yield

@pytest.fixture
def mock_load_model():
    with mock.patch('keras.models.load_model') as mock_load_model:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.6], [0.4], [0.7]])
        mock_load_model.return_value = mock_model
        yield mock_load_model

@pytest.fixture
def mock_mlpreprocessor():
    with mock.patch.object(MLPreprocessor, 'load_pkl', side_effect=[np.array([[1], [0], [1]]), np.array([1, 0, 1])]) as mock_load_pkl, \
         mock.patch.object(MLPreprocessor, 'save_pkl', autospec=True) as mock_save_pkl:
        yield mock_load_pkl, mock_save_pkl

@pytest.fixture
def mock_os():
    with mock.patch('os.makedirs') as mock_makedirs, \
         mock.patch('os.path.exists', return_value=False):
        yield mock_makedirs

def test_predict(mock_dvc_params, mock_load_model, mock_mlpreprocessor, mock_os):
    mock_load_pkl, mock_save_pkl = mock_mlpreprocessor
    predict()

    # Check that load_model was called with the correct path
    mock_load_model.assert_called_once_with("/path/to/trained_model/trained_model.h5")

    # Check that load_pkl was called with the correct paths
    mock_load_pkl.assert_any_call("/path/to/processed_data/url_test.pkl")
    mock_load_pkl.assert_any_call("/path/to/processed_data/label_test.pkl")

    # Check that os.makedirs was called for predictions_path
    mock_os.assert_called_once_with("/path/to/predictions/")

    # Check that save_pkl was called with the correct paths and data
    assert mock_save_pkl.call_count == 2
    np.testing.assert_array_equal(
        mock_save_pkl.call_args_list[0][0][1], np.array([[1], [0], [1]]))
    assert mock_save_pkl.call_args_list[0][0][2] == "/path/to/predictions/label_test_reshaped.pkl"
    
    np.testing.assert_array_equal(
        mock_save_pkl.call_args_list[1][0][1], np.array([[1], [0], [1]]))
    assert mock_save_pkl.call_args_list[1][0][2] == "/path/to/predictions/label_pred_binary.pkl"

    # Ensure that model.predict was called with the correct input
    model_instance = mock_load_model.return_value
    model_instance.predict.assert_called_once_with(np.array([[1], [0], [1]]), batch_size=1000)
