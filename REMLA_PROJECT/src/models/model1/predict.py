import pickle
from keras.models import load_model
import numpy as np


def predict():
    """
    Model prediction
    """
    model = load_model("REMLA_PROJECT\\models\\trained_model.h5")
    with open("REMLA_PROJECT\\data\\processed\\tokenized_data.pkl", "rb") as file:
        tokenized_data = pickle.load(file)
    x_test = tokenized_data["x_test"]
    y_test = tokenized_data["y_test"]

    y_pred = model.predict(x_test, batch_size=1000)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    predictions = {"y_test": y_test, "y_pred_binary": y_pred_binary}

    with open("REMLA_PROJECT\\models\\predictions\\preds.pkl", "wb") as file:
        pickle.dump(predictions, file)


if __name__ == "__main__":
    predict()
