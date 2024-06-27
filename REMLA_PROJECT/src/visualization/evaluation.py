"""
Model predictions evaluation
"""

import os
import seaborn as sns
import dvc.api
from remla_preprocess.pre_processing import MLPreprocessor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score,
)

params = dvc.api.params_show()


def evaluation(y_test=None, y_pred_binary=None):
    """
    Model evaluation
    """
    if y_test is None:
        y_test = MLPreprocessor.load_pkl(
            params["predictions_path"] + "label_test_reshaped.pkl"
        )
    if y_pred_binary is None:
        y_pred_binary = MLPreprocessor.load_pkl(
            params["predictions_path"] + "label_pred_binary.pkl"
        )

    report = classification_report(y_test, y_pred_binary)
    print(f"Classification Report: {report}")

    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print(f"Confusion Matrix: {confusion_mat}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
    sns.heatmap(confusion_mat, annot=True)

    metrics_dict = {
        "accuracy": round(accuracy_score(y_test, y_pred_binary), 5),
        "roc_auc": round(roc_auc_score(y_test, y_pred_binary), 5),
        "f1": round(f1_score(y_test, y_pred_binary), 5),
    }

    if not os.path.exists(params["metrics_path"]):
        os.makedirs(params["metrics_path"])

    MLPreprocessor.save_json(metrics_dict, params["metrics_path"]
                             + "metrics.json", 4)


if __name__ == "__main__":
    evaluation()
