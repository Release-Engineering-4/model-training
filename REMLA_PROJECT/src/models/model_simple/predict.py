"""
Model inference
"""

import os
from keras.models import load_model
import numpy as np
import dvc.api
import pickle
from remla_preprocess.pre_processing import MLPreprocessor

params = dvc.api.params_show()


def predict():
    """
    Model prediction
    """
    model = None
    with open(params["trained_model_path_simple"] + "trained_model_simple.h5", "rb") as f:
        model = pickle.load(f)

    x_test = MLPreprocessor.load_pkl(params["processed_data_path"]
                                     + "url_test.pkl")

    y_test = MLPreprocessor.load_pkl(params["processed_data_path"]
                                     + "label_test.pkl")

    y_pred = model.predict(x_test)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    if not os.path.exists(params["predictions_path_simple"]):
        os.makedirs(params["predictions_path_simple"])

    MLPreprocessor.save_pkl(
        y_test, params["predictions_path_simple"] + "label_test_reshaped_simple.pkl"
    )

    MLPreprocessor.save_pkl(
        y_pred_binary, params["predictions_path_simple"] + "label_pred_binary_simple.pkl"
    )


if __name__ == "__main__":
    predict()
