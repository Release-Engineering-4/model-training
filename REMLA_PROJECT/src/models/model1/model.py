import pickle
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import yaml

def model_definition():
    with open("REMLA_PROJECT\data\processed\char_index.pkl", "rb") as f:
        char_index = pickle.load(f)

    with open("REMLA_PROJECT\configs\params.yaml", "r") as f:
        params = yaml.safe_load(f)

    model = Sequential()
    voc_size = len(char_index.keys())
    print("voc_size: {}".format(voc_size))
    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(len(params['categories'])-1, activation='sigmoid'))

    model.save("REMLA_PROJECT\src\models\model.h5")

if __name__=='__main__':
    model_definition() 