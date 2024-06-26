"""
Model training
"""

import os
from keras.models import load_model
import dvc.api
import pickle
from remla_preprocess.pre_processing import MLPreprocessor

params = dvc.api.params_show()


def train_model():
    """
    Train model
    """
    model = None
    with open(params["model_path_simple"] + "model_simple.h5", "rb") as f:
        model = pickle.load(f)

    x_train = MLPreprocessor.load_pkl(params["processed_data_path"]
                                      + "url_train.pkl")

    y_train = MLPreprocessor.load_pkl(params["processed_data_path"]
                                      + "label_train.pkl")

    x_val = MLPreprocessor.load_pkl(params["processed_data_path"]
                                    + "url_val.pkl")

    y_val = MLPreprocessor.load_pkl(params["processed_data_path"]
                                    + "label_val.pkl")

    model.fit(x_train, y_train)

    if not os.path.exists(params["trained_model_path_simple"]):
        os.makedirs(params["trained_model_path_simple"])

    with open(params["trained_model_path_simple"] + "trained_model_simple.h5", 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
