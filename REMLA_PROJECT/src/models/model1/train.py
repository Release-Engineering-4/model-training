import pickle
from keras.models import load_model
import yaml

def train_model():
    model = load_model("REMLA_PROJECT\models\model.h5")
    with open("REMLA_PROJECT\configs\params.yaml", "r") as f:
        params = yaml.safe_load(f)

    with open("REMLA_PROJECT\data\processed\x_train.pkl", "rb") as f:
        x_train = pickle.load(f)

    with open("REMLA_PROJECT\data\processed\y_train.pkl", "rb") as f:
        y_train = pickle.load(f)

    with open("REMLA_PROJECT\data\processed\x_val.pkl", "rb") as f:
        x_val = pickle.load(f)
    
    with open("REMLA_PROJECT\data\processed\y_val.pkl", "rb") as f:
        y_val = pickle.load(f)

    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])


    hist = model.fit(x_train, y_train,
                    batch_size=params['batch_train'],
                    epochs=params['epoch'],
                    shuffle=True,
                    validation_data=(x_val, y_val)
                    )
    
    model.save("REMLA_PROJECT\models\\trained_model.h5")

if __name__=="__main__":
    train_model() 