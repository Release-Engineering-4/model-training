"""
Preprocess raw data
"""

import os
from remla_preprocess.pre_processing import MLPreprocessor
import dvc.api

params = dvc.api.params_show()


def data_preprocessing(train_data=None, test_data=None, val_data=None):
    """
    Tokenize, pad, encode raw data
    """
    processor = MLPreprocessor()
    if train_data is None:
        train_data = MLPreprocessor.load_txt(params["raw_data_path"]
                                             + "train.txt")

    if test_data is None:
        test_data = MLPreprocessor.load_txt(params["raw_data_path"]
                                            + "test.txt")

    if val_data is None:
        val_data = MLPreprocessor.load_txt(params["raw_data_path"]
                                           + "val.txt")

    train_data = processor.split_data_content(train_data)
    test_data = processor.split_data_content(test_data)
    val_data = processor.split_data_content(val_data)

    trained_data = processor.tokenize_pad_encode_data(train_data,
                                                      test_data,
                                                      val_data)

    if not os.path.exists(params["processed_data_path"]):
        os.makedirs(params["processed_data_path"])

    if not os.path.exists(params["tokenizer_path"]):
        os.makedirs(params["tokenizer_path"])

    for key, value in trained_data.items():
        if key in ("tokenizer", "char_index"):
            MLPreprocessor.save_pkl(value, params["tokenizer_path"]
                                    + key
                                    + ".pkl")
        else:
            MLPreprocessor.save_pkl(value, params["processed_data_path"]
                                    + key
                                    + ".pkl")


if __name__ == "__main__":
    data_preprocessing()
