import time
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import logging

from models.model_interface import ModelInterface
from utils.dataset_builder import DatasetBuilder
from . import get_prompt

def run_exp3(model: ModelInterface, dataset_builder: DatasetBuilder):
    dataset = dataset_builder.build_step_number_dataset()

    tot_latency = 0

    prompt = get_prompt(exp_index=3, model_instance=model)
    labels = []
    responses = []

    for sample in tqdm(dataset):
        images = []
        images.extend([sample["start_frame"]])
        images.extend([sample["end_frame"]])
        images.extend([sample["correct_page"]])
        images.extend([sample["next_page"]])

        label = sample["step_number"]

        start_time = time.time()
        response = model.predict(prompt, images)
        tot_latency += time.time() - start_time

        labels.append(label)
        if type(response) == list:
            response = response[0] # For Qwen-like models

        logging.info(f"Label: {label} - Response: {response}")
        
        if response.isnumeric():
            responses.append(int(response))
        else:
            # Try to extract a number in the response, default to zero-prediction
            responses.append(extract_number(response))

    label_set = list((labels))

    result = {
        "experiment": 3,
        "accuracy": accuracy_score(labels, responses),
        "precision_micro": precision_score(labels, responses, labels=label_set, average="micro"),
        "recall_micro": recall_score(labels, responses, labels=label_set, average="micro"),
        "f1_micro": f1_score(labels, responses, labels=label_set, average="micro"),
        "precision_weighted": precision_score(labels, responses, labels=label_set, average="weighted"),
        "recall_weighted": recall_score(labels, responses, labels=label_set, average="weighted"),
        "f1_weighted": f1_score(labels, responses, labels=label_set, average="weighted"),
        "latency": tot_latency,
        "avg_per_sample_latency": tot_latency / len(dataset)
    }

    return result


def extract_number(output_string):
    match = re.search(r'\d+', output_string)
    return int(match.group()) if match else 0