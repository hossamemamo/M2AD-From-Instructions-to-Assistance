# M2AD-From-Instructions-to-Assistance
This repository contains the code to replicate the experiments on the M2AD Dataset. To start running them, simply configure which models you would like to test and which experiments in the "experiments.yaml" file under the "configs/" folder, and run:
 	`python run.py`
## Before starting
1. Make sure to install all dependencies from the requirements file
2. If you wish to run the experiments on the MolMo model, you first need to apply a (very simple) fix as seen in https://huggingface.co/allenai/Molmo-7B-D-0924/discussions/41
   1. Run the "run.py" script once with just the MolMo model uncommented in the "experiments.yaml" configuration file
   2. The script will return an error after downloading the model and trying to run the experiments
   3. Now that the model is downloaded you can apply the fix, either running `python molmo_fix.py` or manually.
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
    ├── utils/                   # Utility functions
    │   ├── data_loader.py       # Dataset loading and preprocessing
    │   ├── reporting.py         # Results file writing
    │   ├── logging.py           # Logging setup
    │   └── ...           
    └── run.py                   # Main script to orchestrate experiments
