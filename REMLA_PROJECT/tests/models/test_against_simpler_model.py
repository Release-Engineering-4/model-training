import os
import pickle
from sklearn.metrics import accuracy_score

def test_against_a_simle_model_as_baseline():
    # Everytime a model is trained, the accuracy score is saved in a pickle file.
    # We will compare the accuracy score of the model with the simple model.
    label_pred_binary = None
    label_test_reshaped = None

    label_pred_binary_simple = None
    label_test_reshaped_simple = None

    with open("REMLA_PROJECT/models/predictions/label_pred_binary.pkl", "rb") as f:
        label_pred_binary = pickle.load(f)
    
    with open("REMLA_PROJECT/models/predictions/label_test_reshaped.pkl", "rb") as f:
        label_test_reshaped = pickle.load(f)
    
    with open("REMLA_PROJECT/models/predictions_simple/label_pred_binary_simple.pkl", "rb") as f:
        label_pred_binary_simple = pickle.load(f)

    with open("REMLA_PROJECT/models/predictions_simple/label_test_reshaped_simple.pkl", "rb") as f:
        label_test_reshaped_simple = pickle.load(f)
    print(f"Simple accuracy score is:{accuracy_score(label_test_reshaped, label_pred_binary)}")
    assert accuracy_score(label_test_reshaped, label_pred_binary) >= accuracy_score(label_test_reshaped_simple, label_pred_binary_simple)    