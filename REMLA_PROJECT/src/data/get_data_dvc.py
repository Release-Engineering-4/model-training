import pandas as pd

# If we want to get the original data with dvc from a cloud provider like S3 or Google Drive
# import data into data/raw/external using dvc
# Then run this script first in the pipeline and change the "file_name" accordingly


def get_data_dvc():
    """
    Get raw data from cloud provider
    """
    with open("REMLA_PROJECT\\data\\external\\train", "r", encoding="utf-8") as file:
        raw_data_train = [line.strip() for line in file.readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in raw_data_train]
    raw_y_train = [line.split("\t")[0] for line in raw_data_train]

    with open("REMLA_PROJECT\\data\\external\\test", "r", encoding="utf-8") as file:
        raw_data_test = [line.strip() for line in file.readlines()]
    raw_x_test = [line.split("\t")[1] for line in raw_data_test]
    raw_y_test = [line.split("\t")[0] for line in raw_data_test]

    with open("REMLA_PROJECT\\data\\external\\validation", "r", encoding="utf-8") as file:
        raw_data_validation = [line.strip() for line in file.readlines()]
    raw_x_val = [line.split("\t")[1] for line in raw_data_validation]
    raw_y_val = [line.split("\t")[0] for line in raw_data_validation]

    df_train = pd.DataFrame({"label": raw_y_train, "url": raw_x_train})
    df_train.to_csv("REMLA_PROJECT\\data\\raw\\train_data.csv", index=False)

    df_test = pd.DataFrame({"label": raw_y_test, "url": raw_x_test})
    df_test.to_csv("REMLA_PROJECT\\data\\raw\\test_data.csv", index=False)

    df_validation = pd.DataFrame({"label": raw_y_val, "url": raw_x_val})
    df_validation.to_csv("REMLA_PROJECT\\data\\raw\\validation_data.csv", index=False)


if __name__ == "__main__":
    get_data_dvc()
