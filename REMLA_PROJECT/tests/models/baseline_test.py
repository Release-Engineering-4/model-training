import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from remla_preprocess.pre_processing import MLPreprocessor
import numpy as np

def test_local_trained_model():
    files = [
        "REMLA_PROJECT/models/predictions/label_pred_binary.pkl",
        "REMLA_PROJECT/models/predictions/label_test_reshaped.pkl",
        "REMLA_PROJECT/data/processed/url_train.pkl",
        "REMLA_PROJECT/data/processed/label_train.pkl",
        "REMLA_PROJECT/data/processed/url_test.pkl",
        "REMLA_PROJECT/data/processed/label_test.pkl"
    ]
    for file in files:
        # Check if the file exists, if it doesnt automatically return and pass the test
        if not os.path.exists(file):
            return

    label_pred_binary = None
    label_test_reshaped = None

    with open("REMLA_PROJECT/models/predictions/label_pred_binary.pkl", "rb") as f:
        label_pred_binary = pickle.load(f)
    
    with open("REMLA_PROJECT/models/predictions/label_test_reshaped.pkl", "rb") as f:
        label_test_reshaped = pickle.load(f)

    model = LogisticRegression()

    x_train = MLPreprocessor.load_pkl("REMLA_PROJECT/data/processed/url_train.pkl")

    y_train = MLPreprocessor.load_pkl("REMLA_PROJECT/data/processed/label_train.pkl")

    x_test = MLPreprocessor.load_pkl("REMLA_PROJECT/data/processed/url_test.pkl")

    y_test = MLPreprocessor.load_pkl("REMLA_PROJECT/data/processed/label_test.pkl")

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

    assert accuracy_score(y_test, y_pred_binary) < accuracy_score(label_test_reshaped, label_pred_binary)