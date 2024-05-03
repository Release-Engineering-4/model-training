import pickle
from keras.models import load_model
import numpy as np

def predict(): 
    model = load_model("REMLA_PROJECT\models\\trained_model.h5")
    with open("REMLA_PROJECT\data\processed\x_test.pkl", "rb") as f:
        x_test = pickle.load(f)

    y_pred = model.predict(x_test, batch_size=1000) 
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test=y_test.reshape(-1,1)

    with open("REMLA_PROJECT\models\predictions\y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)

    with open("REMLA_PROJECT\models\predictions\y_pred_binary.pkl", "wb") as f:
        pickle.dump(y_pred_binary, f)

if __name__=="__main__":
    predict() 