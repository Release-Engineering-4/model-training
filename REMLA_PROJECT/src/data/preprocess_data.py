from remla_preprocess.pre_processing import MLPreprocessor  
import gdown

if __name__=="__main__":
    train_data = gdown.download_folder(
        id="1DVDHMG4bGD7Y8JYFnDH44urwDoGwXXfZ",
        output="D:\REMLA PROJECT\REMLA_PROJECT\data\\raw",
    )
    # validation_data = MLPreprocessor.get_data_from_gdrive(
    #     "dsadsadsa", "REMLA_PROJECT\data\\raw"
    # )
    # test_data = MLPreprocessor.get_data_from_gdrive(
    #     "dsadsadsa", "REMLA_PROJECT\data\\raw"
    # )
