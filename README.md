# M2AD-From-Instructions-to-Assistance
 
### Directory Structure
    .
    ├── configs/                 # Experiment configurations
    │   └── experiments.yaml     
    ├── dataset/                 # Raw and preprocessed data
    │   └── dataset_ikea.json    
    ├── models/                  # Model wrappers
    │   ├── llava_video.py       
    │   └── llava_onevision.py   
    ├── experiments/             # Experiment code
    │   ├── exp1.py              
    │   ├── exp2.py              
    │   └── exp3.py              
    ├── results/                 # Raw experiment results (JSON)
    ├── reports/                 # Human-readable reports (HTML)
    ├── utils/                   # Utility functions
    │   ├── data_loader.py       # Dataset loading and preprocessing
    │   ├── reporting.py         # Report generation
    │   └── logging.py           # Logging setup
    └── run.py                   # Main script to orchestrate experiments
