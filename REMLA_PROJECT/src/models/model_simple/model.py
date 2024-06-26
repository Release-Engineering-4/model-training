"""
Define model architecture
"""

import os
import dvc.api
from sklearn.tree import DecisionTreeClassifier
import pickle
from remla_preprocess.pre_processing import MLPreprocessor

params = dvc.api.params_show()


def model_definition():
    """
    Define model
    """
    model = DecisionTreeClassifier()

    if not os.path.exists(params["model_path_simple"]):
        os.makedirs(params["model_path_simple"])

    with open(params["model_path_simple"] + "model_simple.h5", 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    model_definition()
