import os
from datasets import Dataset
from huggingface_hub import login

login(os.environ.get("HUGGINGFACE_TOKEN"))


def gen_data(data):
    """
    Upload datasets to huggingface
    """
    for line in data:
        yield {"label": line.strip().split("\t")[0], "url": line.strip().split("\t")[1]}


if __name__ == "__main__":

    with open("dataset_file_path.txt", "r", encoding="utf-8") as file:
        raw_data = file.readlines()
    dataset = Dataset.from_generator(lambda: gen_data(raw_data))
    dataset.push_to_hub("Razvan27/remla_phishing_url", split="validation")
