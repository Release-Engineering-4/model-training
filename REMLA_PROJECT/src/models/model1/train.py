import os
from keras.models import load_model
import dvc.api
from remla_preprocess.pre_processing import MLPreprocessor

params = dvc.api.params_show()


def train_model():
    """
    Train model
    """

    model = load_model(params["model_path"] + "model.h5")

    x_train = MLPreprocessor.load_pkl_data(
            params["processed_data_path"] + "url_train.pkl"
        )

    y_train = MLPreprocessor.load_pkl_data(
            params["processed_data_path"] + "label_train.pkl"
        )

    x_val = MLPreprocessor.load_pkl_data(
            params["processed_data_path"] + "url_val.pkl"
        )

    y_val = MLPreprocessor.load_pkl_data(
            params["processed_data_path"] + "label_val.pkl"
        )

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

    if not os.path.exists(params["trained_model_path"]):
        os.makedirs(params["trained_model_path"])

    model.save(params["trained_model_path"] + "trained_model.h5")


if __name__ == "__main__":
    train_model()
