import pickle
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

def eval(): 
    with open("REMLA_PROJECT\src\models\predictions\y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    with open("REMLA_PROJECT\src\models\predictions\y_pred_binary.pkl", "rb") as f:
        y_pred_binary = pickle.load(f)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:',accuracy_score(y_test,y_pred_binary))
    sns.heatmap(confusion_mat,annot=True)

if __name__=='__main__': 
    eval()