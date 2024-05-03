import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize_data():
    """
    Tokenizes data and saves it.
    """
    training_df = pd.read_csv("REMLA_PROJECT\\data\\raw\\train_data.csv")
    testing_df = pd.read_csv("REMLA_PROJECT\\data\\raw\\test_data.csv")
    validation_df = pd.read_csv("REMLA_PROJECT\\data\\raw\\validation_data.csv")

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token="-n-")
    tokenizer.fit_on_texts(
        training_df["url"] + validation_df["url"] + testing_df["url"]
    )
    char_index = tokenizer.word_index
    sequence_length = 200
    x_train = pad_sequences(
        tokenizer.texts_to_sequences(training_df["url"]), maxlen=sequence_length
    )
    x_val = pad_sequences(
        tokenizer.texts_to_sequences(validation_df["url"]), maxlen=sequence_length
    )
    x_test = pad_sequences(
        tokenizer.texts_to_sequences(testing_df["url"]), maxlen=sequence_length
    )

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(training_df["label"])
    y_val = encoder.transform(validation_df["label"])
    y_test = encoder.transform(testing_df["label"])

    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }

    with open("REMLA_PROJECT\\data\\processed\\tokenized_data.pkl", "wb") as file:
        pickle.dump(data, file)

    with open("REMLA_PROJECT\\data\\processed\\char_index.pkl", "wb") as file:
        pickle.dump(char_index, file)


if __name__ == "__main__":
    tokenize_data()
