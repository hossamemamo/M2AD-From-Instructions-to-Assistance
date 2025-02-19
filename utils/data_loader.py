'''
 This script will handle the following data processing tasks:
    1. Load the dataset from the "dataset_ikea.json" file
    2. Download the YouTube videos from the URLs in the dataset
    3. Download the IKEA Instruction manuals from the URLs in the dataset
    4. Save 'dataset_paths.json' file for video/manual lookup
    5. Extract first and last frames from each video
    6. Save frames and 'frames_paths.json' file for experiments
    7. Work as API for data loading

* Add another method "load_dataloader" to be called instead of the "init" method, that checks for the existence of the dataset and the directories
and enables the API if it finds them. Else, it will call the "init" method to create the dataset and directories.
At this point the init_dataloader method can be made private, so that it can only be called from the load_dataloader method.
'''

import json
import os
import requests
import logging
from typing import List, Dict

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
    init_dataloader(dataset_path="dataset/ikea_dataset.json")
        Downloads videos and manuals and prepares the frames for the experiments
    """
    def __init__(self):
        self.dataset_path = None
        self.base_dir = None
        self.videos_dir = None
        self.manuals_dir = None
        self.tmp_index = -1
        self.loaded = False
        self.mapped_dataset = None
        self.loaded = False

    def init_dataloader(self, 
                    dataset_path="../dataset/ikea_dataset.json", 
                    output_path="../Data/Scraped-Dataset/", 
                    frames_output_path="../Data/Frames/", 
                    pdf_images_output_path="../Data/Pages/"
                    ):
        self.dataset_path = dataset_path
        self.base_dir = output_path
        self.frames_dir = frames_output_path
        self.pdf_images_dir = pdf_images_output_path
        
        with open(self.dataset_path) as file:
            data = json.load(file)

        # Define directories for saving videos and PDFs
        self.videos_dir = os.path.join(base_dir, "Videos")
        self.manuals_dir = os.path.join(base_dir, "Manuals")

        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(manuals_dir, exist_ok=True)

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.pdf_images_dir, exist_ok=True)

        logging.info("Downloading videos and manuals...")
        dataset = []

        for i, obj in enumerate(tqdm(data)):
            if len(obj["annotations"]) == 0:
                continue
            
            self.tmp_index = i

            downloaded_video_paths = self.__scrape_videos(obj["video_url"]) # Download the videos from YouTube
            concatenated_video_path, concatenated_clip = self.__concatenate_videos(downloaded_video_paths) # Concatenate the videos if necessary
            dataset_slice = self.__slice_videos(concatenated_video_path, concatenated_clip, obj["segments"], obj["pdf"]) # Re-slice them for 1:1 correspondence with manuals
            self.__clear_tmp_clip(obj["segments"], concatenated_video_path) # Delete any remaining temp files

            dataset.extend(dataset_slice)

            del concatenated_clip

        self.__save_dataset_paths(dataset)
        self.mapped_dataset = self.__compute_video_manual_mapping(dataset, data) # Map local paths with annotations

        # Frame extraction
        self.__extract_frames()

        # Enable API
        self.loaded = True

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

    def __compute_video_manual_mapping(self, dataset_paths, data) -> List[Dict[str, str, Dict[int, List[Dict[int, int, str, int]]]]]:
        """Maps the downloaded videos and manuals to the annotations in the dataset.

        Parameters
        ----------
        dataset_paths : List[Dict[str, str]]
            A list of dictionaries containing the paths to the videos and manuals.
        
        Returns
        -------
        List[Dict[str, str, Dict[int, List[Dict[int, int, str, int]]]]
            A list of dictionaries containing the paths to the videos and manuals and the list of annotations for this segment.
        """
        mapped_dataset = []
        index = 0
        for entry in data: # For each set of paths
            for annotated_segment in entry["annotations"]:
                mapped_dataset.append({
                    "video": dataset_paths[index]["video"],
                    "manual": dataset_paths[index]["manual"],
                    "annotations": annotated_segment["segment_annotations"]
                })
                index += 1
        return mapped_dataset

    def __extract_frames(self):
        """Extracts the first and last frames from each video and saves them to the frames directory, 
        updating the annotations with the paths to the frames.
        """
        for entry in self.mapped_dataset: # Each of these is a video-manual-annotations triplet
            timestamps = []
            video_path = entry["video"]
            video_id = video_path.split("/")[-1]
            for annotation in entry["annotations"]:
                timestamps.extend(select_n_frames(annotation["start_time"], annotation["end_time"], 2)) # TODO Parametrize n with settings
            
            frames = __extract_frames_bulk(video_path, timestamps) # Returns list of PIL images
            paths = __save_frames_paths(video_id, frames) # Returns list of lists of paths to the frames

            assert len(paths) == len(entry["annotations"]), "Number of extracted frames doesn't match the number of annotations"

            for annotation in entry["annotations"]:
                frames_paths = paths.pop(0)
                annotation["start_frame"] = frames_paths[0]
                annotation["end_frame"] = frames_paths[1]

    def __select_n_frames(self, start_time, end_time, n=2) -> List[int]:
        """Selects n timestamps between start_time and end_time.

        Parameters
        ----------
        start_time : float
            The start time of the video segment.
        
        end_time : float
            The end time of the video segment.

        n : int
            The number of timestamps to select.

        Returns
        -------
        List[int]
            A list of n timestamps between start_time and end_time.
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        
        frames = [round(start_time + i * (end_time - start_time) / (n - 1)) for i in range(n)]
        
        if len(frames) < n:
            logger.warning(f"Expected {n} frames, but only {len(frames)} were selected.")
        
        return frames

    def __extract_frames_bulk(self, video_path, timestamps) -> List[PIL.Image]:
        """Extracts frames from a video at the specified timestamps and returns them as PIL images.

        Parameters
        ----------
        video_path : str
            The path to the video file.
        
        timestamps : List[float]
            A list of timestamps at which to extract the frames.

        Returns
        -------
        List[PIL.Image]
            A list of PIL images representing the extracted frames.
        """
        reader = imageio.get_reader(video_path, 'ffmpeg')
        
        meta_data = reader.get_meta_data() # Get frames per second (fps) and duration using the video metadata
        fps = meta_data['fps']
        duration = meta_data['duration']  # Duration of video
        
        frames = []  # List to store the extracted frames as PIL images
        
        for timestamp in timestamps:
            frame_number = int(fps * timestamp)
            
            # Check if the frame number exceeds the total number of frames
            if timestamp >= duration:
                frame_number = int(fps * duration) - 1  # Set to last frame if beyond video length
            
            try:
                # Seek and read the specified frame
                frame = reader.get_data(frame_number)
                # Convert the frame from RGB to a PIL Image and append to the list
                pil_image = Image.fromarray(frame)
                frames.append(pil_image)
            except IndexError:
                print(f"Failed to read frame at timestamp: {timestamp:.2f}, {frame_number}, for video: {video_path}")
    
        # Close the reader
        reader.close()
        
        return frames

    def __save_frames_paths(self, video_id, frames) -> List[List[str]]:
        """Saves the extracted frames to the frames directory and returns their paths.

        Parameters
        ----------
        video_id : str
            The ID of the video.
        frames : List[PIL.Image]
            A list of PIL images representing the extracted frames.

        Returns
        -------
        List[List[str]]
            A list of lists containing the paths to the extracted frames.
        """
        pattern = r'object_(\d+)_(video_segment|instruction_manual)_(\d+)\.(mp4|pdf)'
        obj_id, seg_id = __extract_x_y(video_id)
        folder_name = f"object_{obj_id}_segment_{seg_id}"

        save_dir = os.path.join(self.frames_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        paths = []
        current_pair = []

        for ix, frame in enumerate(frames):
            # annotation_ix_frame_(start|end).jpg
            frame_filename = f"annotation_{ix//2 + 1}_frame"
            frame_filename += "_start.jpg" if ix % 2 == 0 else "_end.jpg"
            
            frame_path = os.path.join(save_dir, frame_filename)
            # save PIL image to dir
            frame.save(frame_path)
            
            current_pair.append(frame_path)
            
            if len(current_pair) == 2:
                paths.append(current_pair)
                current_pair = []
        return paths

    def __extract_x_y(file_path):
        file_name = Path(file_path).name
        match = re.search(pattern, file_name)
        if match:
            x = int(match.group(1))
            y = int(match.group(3))
            return (x, y)
        else:
            raise ValueError(f"Filename '{file_name}' doesn't match the expected pattern.")