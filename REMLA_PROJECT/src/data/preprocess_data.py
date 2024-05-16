import os
from remla_preprocess.pre_processing import MLPreprocessor
import dvc.api

params = dvc.api.params_show()


def data_preprocessing():
    processor = MLPreprocessor()

    train_data = processor.split_data_content(
        MLPreprocessor.load_txt_data(params["raw_data_path"] + "train.txt")
    )

    test_data = processor.split_data_content(
        MLPreprocessor.load_txt_data(params["raw_data_path"] + "test.txt")
    )

    val_data = processor.split_data_content(
        MLPreprocessor.load_txt_data(params["raw_data_path"] + "val.txt")
    )

    trained_data = processor.tokenize_pad_encode_data(train_data, test_data, val_data)

    if not os.path.exists(params["processed_data_path"]):
        os.makedirs(params["processed_data_path"])

    if not os.path.exists(params["tokenizer_path"]):
        os.makedirs(params["tokenizer_path"])

    for key, value in trained_data.items():
        if key == "tokenizer" or key == "char_index":
            MLPreprocessor.save_data_as_pkl(
                value, params["tokenizer_path"] + key + ".pkl"
            )
        else:
            MLPreprocessor.save_data_as_pkl(
                value, params["processed_data_path"] + key + ".pkl"
            )


if __name__ == "__main__":
    data_preprocessing()
