import time
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

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
        if response.isnumeric():
            responses.append(int(response))
        else:
            # Try to extract a number in the response, default to zero-prediction
            responses.append(extract_number(response))

    label_set = set(labels)

    result = {
        "experiment": 3,
        "accuracy": accuracy_score(labels, responses, labels=label_set),
        "precision": precision_score(labels, responses, labels=label_set),
        "recall": recall_score(labels, responses, labels=label_set),
        "f1": f1_score(labels, responses, labels=label_set),
        "latency": tot_latency,
        "avg_per_sample_latency": tot_latency / len(dataset)
    }

    return result


def extract_number(output_string):
    match = re.search(r'\d+', output_string)
    return int(match.group()) if match else 0