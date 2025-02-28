import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import logging

from models.model_interface import ModelInterface
from utils.dataset_builder import DatasetBuilder
from . import get_prompt

def run_exp1(model: ModelInterface, dataset_builder: DatasetBuilder):
    dataset = dataset_builder.build_completion_dataset()

    tot_latency = 0

    prompt = get_prompt(exp_index=1, model_instance=model)
    labels = []
    responses = []

    flag = 0

    for sample in tqdm(dataset):
        step_no = sample["step_number"]

        prompt = prompt.format(step_number=step_no, img_placeholder="{img_placeholder}")

        images = []
        images.extend([sample["frame"]])
        images.extend([sample["page"]])

        label = sample["label"]

        start_time = time.time()
        response = model.predict(prompt, images)
        tot_latency += time.time() - start_time

        labels.append(label)
        if response.isnumeric() and (int(response) == 0 or int(response) == 1):
            responses.append(int(response))
        else:
            # Default to zero-prediction
            logger.warning(f"Non-numeric prediction detected, defaulting to zero.")
            responses.append(0)

    result = {
        "experiment": 1,
        "accuracy": accuracy_score(labels, responses),
        "precision": precision_score(labels, responses),
        "recall": recall_score(labels, responses),
        "f1": f1_score(labels, responses),
        "latency": tot_latency,
        "avg_per_sample_latency": tot_latency / len(dataset)
    }

    return result