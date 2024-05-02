from datasets import Dataset
from huggingface_hub import login
import os 
login(os.environ.get("HUGGINGFACE_TOKEN"))

def gen_data(data):
    for line in data: 
        yield {"label": line.strip().split("\t")[0], "url": line.strip().split("\t")[1]}


if __name__=="__main__":

    with open("dataset_file_path.txt", "r") as f:
            data = f.readlines()
    dataset = Dataset.from_generator(lambda: gen_data(data)) 
    dataset.push_to_hub("Razvan27/remla_phishing_url", split="validation")