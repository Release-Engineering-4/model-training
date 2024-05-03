import pandas as pd
# In case we want to get the original data with dvc from a cloud provider, e.g., GCS, S3, Google Drive, etc
# import data into data/raw/external using dvc
# Then run this script first in the pipeline and change the "file_name" accordingly 

def get_data_dvc():
    train = [line.strip() for line in open("REMLA_PROJECT\data\external\\file_name", "r").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open("REMLA_PROJECT\data\external\\file_name", "r").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    val=[line.strip() for line in open("REMLA_PROJECT\data\external\\file_name", "r").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]
    raw_y_val=[line.split("\t")[0] for line in val]

    df_train = pd.DataFrame({"label": raw_y_train, "url": raw_x_train})
    df_train.to_csv("REMLA_PROJECT\data\\raw\\train_data.csv", index=False)

    df_test = pd.DataFrame({"label": raw_y_test, "url": raw_x_test})
    df_test.to_csv("REMLA_PROJECT\data\\raw\\test_data.csv", index=False)

    df_validation = pd.DataFrame({"label": raw_y_val, "url": raw_x_val})
    df_validation.to_csv("REMLA_PROJECT\data\\raw\\validation_data.csv", index=False)

if __name__ == "__main__":
    get_data_dvc()