import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


def tokenize_data():
    """
    Tokenizes data and saves it.
    """
    training_df = pd.read_csv(
        "REMLA_PROJECT/data/raw/train_data.csv",
        dtype={"label": str, "url": str}
    )
    testing_df = pd.read_csv(
        "REMLA_PROJECT/data/raw/test_data.csv",
        dtype={"label": str, "url": str}
    )
    validation_df = pd.read_csv(
        "REMLA_PROJECT/data/raw/validation_data.csv",
        dtype={"label": str, "url": str},
    )

    training_df = training_df[["label", "url"]]
    testing_df = testing_df[["label", "url"]]
    validation_df = validation_df[["label", "url"]]

    raw_x_train, raw_y_train = training_df["url"].values,
    training_df["label"].values
    raw_x_test, raw_y_test = testing_df["url"].values,
    testing_df["label"].values
    raw_x_val, raw_y_val = validation_df["url"].values,
    validation_df["label"].values

    tokenizer = Tokenizer(lower=True, char_level=True,
                          oov_token="-n-")
    tokenizer.fit_on_texts(raw_x_train.tolist() +
                           raw_x_val.tolist() +
                           raw_x_test.tolist())
    char_index = tokenizer.word_index
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train),
                            maxlen=200)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val),
                          maxlen=200)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test),
                           maxlen=200)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }

    with open("REMLA_PROJECT/data/processed/tokenized_data.pkl", "wb") as file:
        pickle.dump(data, file)

    with open("REMLA_PROJECT/data/processed/char_index.pkl", "wb") as file:
        pickle.dump(char_index, file)


if __name__ == "__main__":
    tokenize_data()
