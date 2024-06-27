"""
Define model architecture
"""

import os
import dvc.api
from keras.models import Sequential
from keras.layers import (Embedding, Conv1D,
                          MaxPooling1D, Flatten,
                          Dense, Dropout)
from remla_preprocess.pre_processing import MLPreprocessor

params = dvc.api.params_show()


def model_definition(char_index=None):
    """
    Define model
    """
    if char_index is None:
        char_index = MLPreprocessor.load_pkl(params["tokenizer_path"]
                                             + "char_index.pkl")

    model = Sequential()
    voc_size = len(char_index.keys())
    model.add(Embedding(voc_size + 1, 50,
                        input_length=params["max_input_length"]))

    model.add(Conv1D(128, 3, activation="tanh"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation="tanh", padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation="tanh", padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation="tanh", padding="same"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation="tanh", padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation="tanh", padding="same"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation="tanh", padding="same"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(len(params["categories"]) - 1, activation="sigmoid"))

    if not os.path.exists(params["model_path"]):
        os.makedirs(params["model_path"])

    model.save(params["model_path"] + "model.h5")


if __name__ == "__main__":
    model_definition()
