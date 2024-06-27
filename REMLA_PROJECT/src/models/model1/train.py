"""
Model training
"""

import os
from keras.models import load_model
import dvc.api
from remla_preprocess.pre_processing import MLPreprocessor

params = dvc.api.params_show()


def train_model(model=None, x_train=None, y_train=None, x_val=None, y_val=None, custom_params=None):
    """
    Train model
    """
    if model is None:
        model = load_model(params["model_path"] + "model.h5")

    if x_train is None:
        x_train = MLPreprocessor.load_pkl(params["processed_data_path"] + "url_train.pkl")

    if y_train is None:
        y_train = MLPreprocessor.load_pkl(params["processed_data_path"] + "label_train.pkl")

    if x_val is None:
        x_val = MLPreprocessor.load_pkl(params["processed_data_path"] + "url_val.pkl")

    if y_val is None:
        y_val = MLPreprocessor.load_pkl(params["processed_data_path"] + "label_val.pkl")

    if custom_params == None:
        model.compile(
            loss=params["loss_function"],
            optimizer=params["optimizer"],
            metrics=["accuracy"],
        )

        model.fit(
            x_train,
            y_train,
            batch_size=params["batch_train"],
            epochs=params["epoch"],
            shuffle=True,
            validation_data=(x_val, y_val),
        )
    else: 
        model.compile(
            loss=custom_params["loss_function"],
            optimizer=custom_params["optimizer"],
            metrics=custom_params["metrics"],
        )

        model.fit(
            x_train,
            y_train,
            batch_size=custom_params["batch_train"],
            epochs=custom_params["epoch"],
            shuffle=True,
            validation_data=(x_val, y_val),
        )

    if not os.path.exists(params["trained_model_path"]):
        os.makedirs(params["trained_model_path"])

    model.save(params["trained_model_path"] + "trained_model.h5")


if __name__ == "__main__":
    train_model()
