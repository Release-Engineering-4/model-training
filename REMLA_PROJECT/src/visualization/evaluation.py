import json
import pickle
import seaborn as sns
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             accuracy_score, 
                             roc_auc_score, 
                             f1_score)

def evaluation():
    """
    Model evaluation
    """
    with open("REMLA_PROJECT/models/predictions/preds.pkl", "rb") as file:
        predictions = pickle.load(file)
    y_test = predictions["y_test"]
    y_pred_binary = predictions["y_pred_binary"]

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print(f"Classification Report: {report}")

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print(f"Confusion Matrix: {confusion_mat}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
    sns.heatmap(confusion_mat, annot=True)

    metrics_dict = {
    "accuracy": round(accuracy_score(y_test, y_pred_binary),5),
    "roc_auc": round(roc_auc_score(y_test, y_pred_binary),5),
    "f1": round(f1_score(y_test, y_pred_binary),5)
    }

    with open("REMLA_PROJECT/reports/metrics.json", "w", encoding="utf-8") as json_file:
        json.dump(metrics_dict, json_file, indent=4)


if __name__ == "__main__":
    evaluation()
