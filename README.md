# M2AD-From-Instructions-to-Assistance
This repository contains the code to replicate the experiments on the M2AD Dataset. To start running them, simply configure which models you would like to test and which experiments in the "experiments.yaml" file under the "configs/" folder, and run:
 	`python run.py`
## Before starting
1. Make sure to install all dependencies from the requirements file
2. If you would like to evaluate the MolMo model, please make sure to implement the small fix (2 lines to change) seen in https://huggingface.co/allenai/Molmo-7B-D-0924/discussions/41 (This is very important, otherwise the model will NOT run)
### Directory Structure
    .
    ├── configs/                 # Experiment configurations
    │   └── experiments.yaml     
    ├── dataset/                 # Raw and preprocessed data
    │   └── dataset_ikea.json    
    ├── models/                  # Model wrappers
    │   ├── llava_video.py       
    │   ├── llava_onevision.py   
    │   └── ...   
    ├── experiments/             # Experiment code
    │   ├── exp1.py              
    │   ├── exp2.py              
    │   └── exp3.py              
    ├── results/                 # Raw experiment results (JSON)
    ├── reports/                 # Human-readable reports (HTML)
    ├── utils/                   # Utility functions
    │   ├── data_loader.py       # Dataset loading and preprocessing
    │   ├── reporting.py         # Results file writing
    │   ├── logging.py           # Logging setup
    │   └── ...           
    └── run.py                   # Main script to orchestrate experiments
