import pickle
from keras.models import load_model
import yaml


def train_model():
    """
    Train model
    """

    model = load_model("REMLA_PROJECT\\models\\model.h5")
    with open("REMLA_PROJECT\\configs\\params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    with open(
        "REMLA_PROJECT\\data\\processed\\tokenized_data.pkl", "rb", encoding="utf-8"
    ) as file:
        tokenized_data = pickle.load(file)

    x_train = tokenized_data["x_train"]
    y_train = tokenized_data["y_train"]
    x_val = tokenized_data["x_val"]
    y_val = tokenized_data["y_val"]

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

    model.save("REMLA_PROJECT\\models\\trained_model.h5")


if __name__ == "__main__":
    train_model()
