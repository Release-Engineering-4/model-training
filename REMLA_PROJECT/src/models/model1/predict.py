"""
Model inference
"""

import os
from keras.models import load_model
import numpy as np
import dvc.api
from remla_preprocess.pre_processing import MLPreprocessor

params = dvc.api.params_show()


def predict(model=None, x_test=None, y_test=None):
    """
    Model prediction
    """
    if model is None:
        model = load_model(params["trained_model_path"] + "trained_model.h5")

    if x_test is None:
        x_test = MLPreprocessor.load_pkl(params["processed_data_path"]
                                        + "url_test.pkl")
    if y_test is None:
        y_test = MLPreprocessor.load_pkl(params["processed_data_path"]
                                        + "label_test.pkl")

    y_pred = model.predict(x_test, batch_size=1000)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    if not os.path.exists(params["predictions_path"]):
        os.makedirs(params["predictions_path"])

    MLPreprocessor.save_pkl(
        y_test, params["predictions_path"] + "label_test_reshaped.pkl"
    )

    MLPreprocessor.save_pkl(
        y_pred_binary, params["predictions_path"] + "label_pred_binary.pkl"
    )


if __name__ == "__main__":
    predict()
