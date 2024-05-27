import pytest
from unittest import mock
import dvc.api
import seaborn as sns
import numpy as np
from remla_preprocess.pre_processing import MLPreprocessor
from src.visualization import evaluation
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score,
)

# Mock DVC params
mock_params = {
    "predictions_path": "/path/to/predictions/",
    "metrics_path": "/path/to/metrics/"
}

@pytest.fixture
def mock_dvc_params():
    with mock.patch.object(dvc.api, 'params_show', return_value=mock_params):
        yield

@pytest.fixture
def mock_mlpreprocessor():
    y_test = np.array([1, 0, 1])
    y_pred_binary = np.array([1, 0, 0])
    with mock.patch.object(MLPreprocessor, 'load_pkl', side_effect=[y_test, y_pred_binary]) as mock_load_pkl, \
         mock.patch.object(MLPreprocessor, 'save_json', autospec=True) as mock_save_json:
        yield mock_load_pkl, mock_save_json

@pytest.fixture
def mock_os():
    with mock.patch('os.makedirs') as mock_makedirs, \
         mock.patch('os.path.exists', return_value=False):
        yield mock_makedirs

@pytest.fixture
def mock_sns():
    with mock.patch.object(sns, 'heatmap', autospec=True) as mock_heatmap:
        yield mock_heatmap

def test_evaluation(mock_dvc_params, mock_mlpreprocessor, mock_os, mock_sns):
    mock_load_pkl, mock_save_json = mock_mlpreprocessor
    evaluation()

    # Check that load_pkl was called with the correct paths
    mock_load_pkl.assert_any_call("/path/to/predictions/label_test_reshaped.pkl")
    mock_load_pkl.assert_any_call("/path/to/predictions/label_pred_binary.pkl")

    # Calculate expected metrics
    y_test = np.array([1, 0, 1])
    y_pred_binary = np.array([1, 0, 0])
    expected_accuracy = round(accuracy_score(y_test, y_pred_binary), 5)
    expected_roc_auc = round(roc_auc_score(y_test, y_pred_binary), 5)
    expected_f1 = round(f1_score(y_test, y_pred_binary), 5)

    # Check that os.makedirs was called for metrics_path
    mock_os.assert_called_once_with("/path/to/metrics/")

    # Check that save_json was called with the correct data
    expected_metrics_dict = {
        "accuracy": expected_accuracy,
        "roc_auc": expected_roc_auc,
        "f1": expected_f1
    }
    mock_save_json.assert_called_once_with(expected_metrics_dict, "/path/to/metrics/metrics.json", 4)

    # Ensure that sns.heatmap was called with the correct confusion matrix
    expected_confusion_matrix = confusion_matrix(y_test, y_pred_binary)
    mock_sns.assert_called_once_with(expected_confusion_matrix, annot=True)
