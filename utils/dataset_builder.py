"""
This class will handle all tasks related to building datasets for the related experiments.

For Experiment 1 (Completion), we will need to build a dataset of the following format:
[
    {
        "frame": PIL.Image,
        "page": PIL.Image,
        "label": int (0/1)
    }
]
Which takes both start and end frames, and only the correct page from the ones extracted by the DataLoader module. Start frames are mapped to 0 (incomplete examples), and end frames are mapped to 1 (complete examples). The dataset must be shuffled at the end.

For Experiment 2 (Relevancy prediction), we will need to build a dataset of the following format:
[
    {
        "frame": PIL.Image,
        "page": PIL.Image,
        "label": int (0/1)
    }
]
Which takes both the correct and next/random page, and only the first frame from the ones extracted by the DataLoader module. Correct pages are mapped to 1, and random pages are mapped to 0. The dataset must be shuffled at the end.

For Experiment 3 (Step number prediction), we will need to build a dataset of the following format:
[
    {
        "frame": PIL.Image,
        "page": PIL.Image,
        "label": int (0-max_step)
    }
]
Which takes both the correct and next page, and both the initial and final frames extracted by the DataLoader module. The label is the assembly step number of the assembly step being performed in the frames. The dataset must be shuffled at the end.
"""
from utils.data_manager import DataManager

import numpy as np
import logging
from PIL import Image

class DatasetBuilder:
    def __init__(self, data_manager:DataManager):
        self.data_manager = data_manager
        self.mapped_dataset = self.data_manager.mapped_dataset

    def build_completion_dataset(self):
        dataset = []
        for video in self.mapped_dataset:
            for annotation in video["annotations"]:
                a = None

    def build_relevancy_dataset(self):
        dataset = []
        pass

    def build_step_number_dataset(self):
        dataset = []
        pass

    def shuffle_dataset(self, dataset):
        pass

