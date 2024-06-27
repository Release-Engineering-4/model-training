import importlib.util
import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import os


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


@pytest.fixture
def mock_data():
    return {
        "x_test": np.random.rand(50, 10),
        "y_test": np.random.randint(0, 2, 50),
    }


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict = MagicMock()
    return model


@pytest.mark.parametrize(
    "x_test, y_test",
    [
        (None, None),
        (
            np.random.rand(50, 10),
            np.random.randint(0, 2, 50),
        ),
    ],
)
def test_predict(mock_data, mock_model, x_test, y_test):
    with patch("dvc.api.params_show") as mock_params_show, patch(
        "remla_preprocess.pre_processing.MLPreprocessor.load_pkl"
    ) as mock_load_pkl, patch("os.path.exists", return_value=True), patch(
        "os.makedirs"
    ):

        mock_params = {
            "trained_model_path": "REMLA_PROJECT/models/trained_model/",
            "processed_data_path": "REMLA_PROJECT/data/processed/",
            "predictions_path": "REMLA_PROJECT/predictions/",
        }
        mock_params_show.return_value = mock_params

        mock_load_pkl.side_effect = lambda arg: (
            mock_data["x_test"] if "url_test.pkl" in arg else mock_data["y_test"]
        )

        predict_model(model=mock_model, x_test=x_test, y_test=y_test)

        if x_test is None:
            mock_model.predict.assert_called_once_with(
                mock_data["x_test"], batch_size=1000
            )
        else:
            mock_model.predict.assert_called_once_with(x_test, batch_size=1000)

        assert os.path.exists(
            mock_params["predictions_path"] + "label_test_reshaped.pkl"
        )
        assert os.path.exists(mock_params["predictions_path"] + "label_pred_binary.pkl")
