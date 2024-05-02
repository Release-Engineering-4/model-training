import pickle
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences


def tokenize_data():
    training_df = pd.read_csv("REMLA_PROJECT\data\\raw\\train_data.csv")
    testing_df = pd.read_csv("REMLA_PROJECT\data\\raw\\test_data.csv")
    validation_df = pd.read_csv("REMLA_PROJECT\data\\raw\\validation_data.csv")
    
    raw_x_train = training_df["url"]
    raw_y_train = training_df["label"]

    raw_x_test = testing_df["url"]
    raw_y_test = testing_df["label"]

    raw_x_val = validation_df["url"]
    raw_y_val = validation_df["label"]

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    sequence_length=200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    save_data(x_train, y_train, x_val, y_val, x_test, y_test, char_index)

def save_data(x_train, y_train, x_val, y_val, x_test, y_test, char_index):
    with open("REMLA_PROJECT\data\processed\x_train.pkl", "wb") as f:
        pickle.dump(x_train, f)
    with open("REMLA_PROJECT\data\processed\y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open("REMLA_PROJECT\data\processed\x_val.pkl", "wb") as f:
        pickle.dump(x_val, f)
    with open("REMLA_PROJECT\data\processed\y_val.pkl", "wb") as f:
        pickle.dump(y_val, f)
    with open("REMLA_PROJECT\data\processed\x_test.pkl", "wb") as f:
        pickle.dump(x_test, f)
    with open("REMLA_PROJECT\data\processed\y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)
    with open("REMLA_PROJECT\data\processed\char_index.pkl", "wb") as f:
        pickle.dump(char_index, f)


if __name__=='__main__':
    tokenize_data() 