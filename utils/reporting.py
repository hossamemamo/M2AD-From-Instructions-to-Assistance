import json
import os
from typing import List, Dict

def save_raw_results(results: List[Dict], output_path: str):
    """
    Saves partial results to a JSON file.
    
    Args:
        results (List[Dict]): A list of results (each result is a dictionary).
        output_path (str): Path to the output JSON file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load existing results if the file exists
    #if os.path.exists(output_path):
    #    with open(output_path, "r") as f:
    #        existing_results = json.load(f)
    #else:
    #    existing_results = []
    
    # Append new results
    #existing_results.extend(results)
    
    # Save updated results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)