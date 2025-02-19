'''
 This script will handle the following data processing tasks:
    1. Load the dataset from the "dataset_ikea.json" file
    2. Download the YouTube videos from the URLs in the dataset
    3. Download the IKEA Instruction manuals from the URLs in the dataset
    4. Save 'dataset_paths.json' file for video/manual lookup
    5. Extract first and last frames from each video
    6. Save frames and 'frames_paths.json' file for experiments
    7. Work as API for data loading
'''

import json
import os
import requests
import logging

from pytubefix import YouTube
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import concatenate_videoclips, VideoFileClip

class DataLoader:
    """
    This class is used to handle all data preparation tasks for the experiments presented in this work.

    Attributes
    ----------
    var_name : type
        a formatted string to print out what the animal says

    Methods
    -------
    load_dataset(dataset_path="dataset/ikea_dataset.json")
        Downloads videos and manuals and prepares the frames for the experiments
    """
    def __init__(self):
        self.base_dir = None
        self.videos_dir = None
        self.manuals_dir = None
        self.tmp_index = -1

    def load_dataset(self, dataset_path="../dataset/ikea_dataset.json", output_path="../Data/Scraped-Dataset/"):
        # Sample JSON data
        with open(dataset_path) as file:
            data = json.load(file)

        # Define directories for saving videos and PDFs
        self.base_dir = output_path
        self.videos_dir = os.path.join(base_dir, "Videos")
        self.manuals_dir = os.path.join(base_dir, "Manuals")

        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(manuals_dir, exist_ok=True)

        logging.info("Downloading videos and manuals...")
        dataset = []

        for i, obj in enumerate(tqdm(data)):
            if len(obj["annotations"]) == 0:
                continue
            
            self.tmp_index = i

            downloaded_video_paths = self.__scrape_videos(obj["video_url"])
            concatenated_video_path, concatenated_clip = self.__concatenate_videos(downloaded_video_paths)
            dataset_slice = self.__slice_videos(concatenated_video_path, concatenated_clip, obj["segments"], obj["pdf"])
            self.__clear_tmp_clip(obj["segments"], concatenated_video_path)

            dataset.extend(dataset_slice)

        self.__save_dataset_paths(dataset)

    def __scrape_videos(self, video_urls) -> List[str]:
        """Downloads the videos from the given URLs and returns their local paths.

        Parameters
        ----------
        video_urls : List[str]
            A list of URLs to download the videos from.

        Returns
        -------
        List[str]
            A list of local paths to the downloaded videos.
        """
        video_paths = []
        for j, url in enumerate(video_urls):
            video_path = os.path.join(self.videos_dir, f"object_{self.tmp_index}_video_{j}.mp4")
            while True:
                try:
                    self.__download_video(url, video_path)
                except IOError:
                    continue
                break
            video_paths.append(video_path)
        return video_paths

    def __concatenate_videos(self, video_paths) -> VideoFileClip:
        """Concatenates multiple videos into one if necessary.

        Parameters
        ----------
        video_paths : List[str]
            A list of paths to the videos to concatenate.

        Returns
        -------
        VideoFileClip
            A VideoFileClip object representing the concatenated video.
        """
        if len(video_paths) == 1:
            concatenated_video_path = video_paths[0]
            concatenated_clip = VideoFileClip(concatenated_video_path)
        else:
            video_clips = [VideoFileClip(video_path) for video_path in video_paths]
            concatenated_video_path = os.path.join(videos_dir, f"object_{self.tmp_index}_concatenated_video.mp4")
            concatenated_clip = concatenate_videoclips(video_clips, method="compose")
            concatenated_clip.write_videofile(concatenated_video_path, codec='libx264', audio_codec='aac')

            # Delete individual video files after concatenation
            for video_path in video_paths:
                os.remove(video_path)

        return concatenated_video_path, concatenated_clip

    def __slice_videos(self, concatenated_video_path, concatenated_clip, segments, manuals) -> Dict[str, str]:
        """Slices the concatenated video into segments for 1:1 mapping with manuals and downloads the associated instruction manual PDFs.

        Parameters
        ----------
        concatenated_video_path : str
            The path to the concatenated video.
        concatenated_clip : VideoFileClip
            The VideoFileClip object representing the concatenated video.
        segments : List[Dict[int, int, int]]
            A list of dictionaries containing the start and end timestamps of each segment and the manual number.
        manuals : List[str]
            A list of URLs to download the instruction manuals from.

        Returns
        -------
        Dict[str, str]
            A dictionary containing the paths to the sliced videos and the instruction manuals.
        """
        dataset_slice = []
        for sn, segment in enumerate(segments):
            start_time = segment["start_time"]
            end_time = segment["end_time"] if segment["end_time"] != -1 else concatenated_clip.duration
            manual_num = segment["manual_num"]

            video_filename = f"object_{self.tmp_index}_video_segment_{sn}.mp4"
            instruction_manual_filename = f"object_{self.tmp_index}_instruction_manual_{manual_num}.pdf"

            output_video_path = os.path.join(self.videos_dir, video_filename)
            if start_time == 0 and end_time == concatenated_clip.duration:
                os.rename(concatenated_video_path, os.path.join(self.videos_dir, video_filename))
            else:
                ffmpeg_extract_subclip(concatenated_video_path, start_time, end_time, targetname=output_video_path)

            # Download the associated PDF
            manual_name = manuals[manual_num]
            pdf_path = os.path.join(self.manuals_dir, instruction_manual_filename)
            self.download_manual(manual_name, pdf_path)

            # Return the paths to the dataset list
            dataset_slice.append({
                "pdf": pdf_path,
                "video_url": output_video_path
            })

        return dataset_slice

    def __clear_tmp_clip(self, segments, concatenated_video_path):
        """Deletes the temporary concatenated video file.

        Parameters
        ----------
        segments : List[Dict[int, int, int]]
            A list of dictionaries containing the start and end timestamps of each segment and the manual number.
        concatenated_video_path : str
            The path to the concatenated video.
        """
        if len(segments) > 1:
            os.remove(concatenated_video_path)

    def __download_video(self, url, save_path):
        """Downloads a video from a given URL.

        Parameters
        ----------
        url : str
            The URL to download the video from.
        save_path : str
            The path to save the downloaded video to.

        Returns
        -------
        int
            1 if the video was downloaded successfully, 0 otherwise.
        """
        try:
            # Create a YouTube object using the URL
            yt = YouTube(url)
            
            # Get the highest resolution stream available
            stream = yt.streams.get_highest_resolution()

            # Download the video to the specified path
            stream.download(output_path=os.path.dirname(save_path), filename=os.path.basename(save_path))
            return 1
        except Exception as e:
            logging.error(f"Failed to download video from {url}: {e}")
            raise IOError("Failed to download video")
        pass

    def __download_manual(self, url, save_path):
        """Downloads an instruction manual PDF from a given URL.

        Parameters
        ----------
        url : str
            The URL to download the PDF from.
        save_path : str
            The path to save the downloaded PDF to.

        Returns
        -------
        int
            1 if the PDF was downloaded successfully, 0 otherwise.
        """
        try:
            # Send a GET request to the URL
            response = requests.get(url)

            # Raise an exception if the request was unsuccessful
            response.raise_for_status()

            # Write the content of the response to a file
            with open(save_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
            return 1

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download PDF from {url}: {e}")
        pass

    def __save_dataset_paths(self, dataset):
        """Saves the dataset paths to a JSON file.

        Parameters
        ----------
        dataset : List[Dict[str, str]]
            A list of dictionaries containing the paths to the videos and manuals.
        """
        output_json_path = os.path.join(self.base_dir, "dataset_paths.json")
        with open(output_json_path, 'w') as json_file:
            json.dump(dataset, json_file, indent=4)

        logging.info(f"Dataset JSON saved to {output_json_path}")

    def __extract_frames(self):
        a = None

    def __save_frames_paths(self):
        a = None