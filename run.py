import yaml
import logging

from models import load_model
from utils.data_manager import DataManager
from utils.dataset_builder import DatasetBuilder
from utils.logging_manager import setup_logging
from utils.reporting import save_raw_results
from experiments.exp1 import run_exp1
from experiments.exp2 import run_exp2
from experiments.exp3 import run_exp3

def load_config(config_path="configs/experiments.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    setup_logging()

    data_manager = DataManager()
    data_manager.load_dataloader()

def main2():    
    dataset_builder = DatasetBuilder(data_manager)

    experiment_mapping = {
        "exp1": run_exp1,
        "exp2": run_exp2,
        "exp3": run_exp3
    }

    # Run experiments on specified models
    results = []
    logging.info("Beginning experiments...")
    for model_name in config["models"]:
        logging.info(f"Loading model {model_name}")
        model = load_model(model_name)
        model_results = []

        for exp in config["experiments"]:
            if exp not in experiment_mapping:
                raise ValueError(f"Invalid experiment: {exp}")
                break
            experiment = experiment_mapping[exp]

            logging.info(f"Running Experiment {exp} for model {model_name}")
            result = experiment(model, dataset_builder)
            logging.info(f"Completed Experiment {exp} for model {model_name}")
            model_results.append(result)

        del model
        results.append({
            "model": model_name,
            "results": model_results
        })
        logging.info(f"Saving final results for model {model_name}")
        # Save raw results for each model
        save_raw_results(results, output_path="./results/raw_results.json")

    # Save final results and generate report
    save_raw_results(results, output_path="./results/raw_results.json")
    logging.info("Generating report...")
    logging.info("Done!")


if __name__ == "__main__":
    main()