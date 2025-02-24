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

import random
import logging
from PIL import Image
from typing import List, Dict

class DatasetBuilder:
    def __init__(self, data_manager:DataManager):
        self.data_manager = data_manager
        self.mapped_dataset = self.data_manager.mapped_dataset
        random.seed(42)

    def build_completion_dataset(self) -> List[Dict]:
        """Builds the dataset for the completion experiment
        
        Returns
        -------
        List[Dict]
            The dataset built for the completion task.
        """
        dataset = []
        for video in self.mapped_dataset:
            for annotation in video["annotations"]:
                positive_sample = {
                    "frame": Image.open(annotation["end_frame"]),
                    "page": Image.open(annotation["page_path"]),
                    "step_number": int(annotation["label"][:1]),
                    "label": 1
                }

                negative_sample = {
                    "frame": Image.open(annotation["start_frame"]),
                    "page": Image.open(annotation["page_path"]),
                    "step_number": int(annotation["label"][:1]),
                    "label": 0
                }
                dataset.append(positive_sample)
                dataset.append(negative_sample)
        dataset = self.__shuffle_dataset(dataset)
        return dataset

    def build_relevancy_dataset(self) -> List[Dict]:
        """Builds the dataset for the relevancy prediction experiment

        Returns
        -------
        List[Dict]
            The dataset built for the relevancy prediction task.
        """
        dataset = []
        for video in self.mapped_dataset:
            for index, annotation in enumerate(video["annotations"]):
                positive_sample = {
                    "frame": Image.open(annotation["start_frame"]),
                    "page": Image.open(annotation["page_path"]),
                    "label": 1
                }

                # Get a random index that is not the current index and extract the page path from the entry relative to that index
                random_index = random.choice([i for i in range(len(video["annotations"])) if i != index])

                random_page_path = video["annotations"][random_index]["page_path"] # We take a random page as it should be easier to distinguish compared to the next page

                negative_sample = {
                    "frame": Image.open(annotation["start_frame"]),
                    "page": Image.open(random_page_path),
                    "label": 0
                }
                dataset.append(positive_sample)
                dataset.append(negative_sample)
        dataset = self.__shuffle_dataset(dataset)
        return dataset

    def build_step_number_dataset(self) -> List[Dict]:
        """Builds the dataset for the step number prediction experiment

        Returns
        -------
        List[Dict]
            The dataset built for the step number prediction task.
        """
        dataset = []
        for video in self.mapped_dataset:
            for annotation in video["annotations"]:
                entry = {
                    "start_frame": Image.open(annotation["start_frame"]),
                    "end_frame": Image.open(annotation["end_frame"]),
                    "correct_page": Image.open(annotation["page_path"]),
                    "next_page": Image.open(annotation["next_page_path"]),
                    "step_number": int(annotation["label"][:1])
                }
                dataset.append(entry)
        dataset = self.__shuffle_dataset(dataset)
        return dataset

    def shuffle_dataset(self, dataset) -> List[Dict]:
        """Shuffles the dataset

        Parameters
        ----------
        dataset : List[Dict]
            The dataset to be shuffled

        Returns
        -------
        List[Dict]
            The shuffled dataset
        """
        random.shuffle(dataset)
        return dataset
