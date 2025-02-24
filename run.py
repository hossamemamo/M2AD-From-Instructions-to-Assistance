import yaml

from models import load_model
from utils.data_manager import DataManager
from utils.dataset_builder import DatasetBuilder
from utils.logging_manager import setup_logging
from experiments.exp1 import run_exp1
from experiments.exp2 import run_exp2
from experiments.exp3 import run_exp3

def load_config(config_path="./configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    setup_logging()

    data_manager = DataManager()
    data_manager.load_dataloader()

    dataset_builder = DatasetBuilder(data_manager)

    experiment_mapping = {
        "exp1": run_exp1,
        "exp2": run_exp2,
        "exp3": run_exp3
    }

    # Run experiments on specified models
    results = []
    for model_name in config["models"]:
        model = load_model(model_name)
        model_results = []

        for exp in config["experiments"]:
            if exp not in experiment_mapping:
                raise ValueError(f"Invalid experiment: {exp}")
                break
            experiment = experiment_mapping[exp]

            result = experiment(model, dataset_builder)
            model_results.append(result)

        del model
        results.append({
            "model": model_name,
            "results": model_results
        })

        # Save raw results for each model
        a = None

    # Save final results and generate report
    a = None