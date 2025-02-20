import json
import os
import re
import requests
import logging
from typing import List, Dict, Tuple

from pytubefix import YouTube
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import concatenate_videoclips, VideoFileClip
import pypdfium2 as pdfium
import imageio
from pathlib import Path
from PIL import Image

class DataManager:
    """
    This class is used to handle all data preparation tasks for the experiments presented in this work.

    Attributes
    ----------
    dataset_path : str
        The path to the dataset JSON file.
    base_dir : str
        The base directory for saving the downloaded videos and manuals.
    scraping_dir : str
        The directory for saving the scraped videos and manuals.
    videos_dir : str
        The directory for saving the downloaded videos.
    manuals_dir : str
        The directory for saving the downloaded manuals.
    frames_dir : str
        The directory for saving the extracted frames.
    pdf_images_dir : str
        The directory for saving the extracted PDF pages.
    tmp_index : int
        A temporary index used for naming the downloaded files.
    mapped_dataset : List[Dict[str, str, Dict[int, List[Dict[int, int, str, int]]]]
        A list of dictionaries containing the paths to the videos and manuals and the list of annotations for this segment.
    loaded : bool
        A flag indicating whether the data preparation is complete.
    dataset_filename : str
        The name of the file to save the dataset paths.

    Methods
    -------
    load_dataloader(dataset_path, output_path)
        Loads the dataset and checks for cached data.
    __init_dataloader(dataset_path, output_path)
        Initializes the data preparation process.
    __scrape_videos(video_urls)
        Downloads the videos from the given URLs and returns their local paths.
    __concatenate_videos(video_paths)
        Concatenates multiple videos into one if necessary.
    __slice_videos(concatenated_video_path, concatenated_clip, segments, manuals)
        Slices the concatenated video into segments for 1:1 mapping with manuals and downloads the associated instruction manual PDFs.
    __clear_tmp_clip(segments, concatenated_video_path)
        Deletes the temporary concatenated video file.
    __download_video(url, save_path)
        Downloads a video from a given URL.
    __download_manual(url, save_path)
        Downloads an instruction manual PDF from a given URL.
    __save_dataset_to_json(data)
        Saves the dataset to a JSON file.
    __compute_video_manual_mapping(dataset_paths, data)
        Maps the downloaded videos and manuals to the annotations in the dataset.
    __extract_frames()
        Extracts the first and last frames from each video and saves them to the frames directory, updating the annotations with the paths to the frames.
    __extract_pages()
        Extracts pages from the PDFs and saves them to the PDF images directory, updating the annotations with the paths to the pages.
    __select_n_frames(start_time, end_time, n=2)
        Selects n timestamps between start_time and end_time.
    __extract_frames_bulk(video_path, timestamps)
        Extracts frames from a video at the specified timestamps and returns them as PIL images.
    __extract_pages(pdf_path, pages)
        Extracts pages from a PDF at the specified indices and returns them as PIL images.
    __save_frames_paths(video_id, frames)
        Saves the extracted frames to the frames directory and returns their paths.
    __save_pages_paths(pdf_id, pages)
        Saves the extracted pages to the PDF images directory and returns their paths.
    __get_manual_page_boundaries(entry)
        Extracts the min and max page from the annotations of an instruction manual.
    __extract_x_y(file_path)
        Extracts the object and segment IDs from the file path.
    """
    def __init__(self):
        self.dataset_path = None
        self.base_dir = None
        
        self.scraping_dir = None
        self.videos_dir = None
        self.manuals_dir = None
        self.frames_dir = None
        self.pdf_images_dir = None
        
        self.tmp_index = -1

        self._mapped_dataset = None
        self.loaded = False

        self.dataset_filename = "dataset_paths.json"

    @property
    def mapped_dataset(self):
        assert self.loaded, "Data preparation not complete. Load the dataloader first."
        return self._mapped_dataset

    def load_dataloader(self, 
                    dataset_path="../dataset/ikea_dataset.json", 
                    output_path="../Data/"
                    ):
        logging.info("Checking for cached dataset and directories...")
        mapped_dataset_path = os.path.join(output_path, self.dataset_filename)

        if not os.path.exists(mapped_dataset_path):
            logging.info("No cached dataset found. Initializing data preparation...")
            self.__init_dataloader(dataset_path, output_path)
            return

        with open(mapped_dataset_path) as file:
            self._mapped_dataset = json.load(file)

        # Paths to check are 'video' 'manual' and 'start_frame' 'end_frame' 'page_path' 'next_page_path' into 'annotations'
        missing_videos_manuals = False
        missing_frames_pages = False
        for segment in self._mapped_dataset:
            video_path = segment["video"]
            if not os.path.exists(video_path):
                logging.warning(f"Video path {video_path} not found.")
                missing_videos_manuals = True
                break
            
            manual_path = segment["manual"]
            if not os.path.exists(manual_path):
                logging.warning(f"Manual path {manual_path} not found.")
                missing_videos_manuals = True
                break
            
            for annotation in segment["annotations"]:
                if not os.path.exists(annotation["start_frame"]) or not os.path.exists(annotation["end_frame"]):
                    logging.warning(f"Frame paths {annotation['start_frame']} or {annotation['end_frame']} not found.")
                    missing_frames_pages = True
                    break
                
                if not os.path.exists(annotation["page_path"]) or not os.path.exists(annotation["next_page_path"]):
                    logging.warning(f"Page paths {annotation['page_path']} or {annotation['next_page_path']} not found.")
                    missing_frames_pages = True
                    break
            
            if missing_frames_pages:
                break
        
        if missing_videos_manuals:
            logging.info("Missing videos or manuals. Re-initializing data preparation...")
            self.__init_dataloader(dataset_path, output_path)
            return
        
        if missing_frames_pages:
            logging.info("Missing frames or pages. Re-extracting...")
            self.__extract_frames_pages()
            return

        logging.info("All paths found. Data preparation complete.")
        self.loaded = True


    def __init_dataloader(self, dataset_path="../dataset/ikea_dataset.json", output_path="../Data/"):
        self.dataset_path = dataset_path
        self.base_dir = output_path

        # Define directories for saving videos and PDFs
        self.scraping_dir = os.path.join(self.base_dir, "Scraped-Dataset")
        self.videos_dir = os.path.join(self.scraping_dir, "Videos")
        self.manuals_dir = os.path.join(self.scraping_dir, "Manuals")
        self.frames_dir = os.path.join(self.base_dir, "Frames")
        self.pdf_images_dir = os.path.join(self.base_dir, "Pages")

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.scraping_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.manuals_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.pdf_images_dir, exist_ok=True)

        with open(self.dataset_path) as file:
            data = json.load(file)

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

        self._mapped_dataset = self.__compute_video_manual_mapping(dataset, data) # Map local paths with annotations
        self.__extract_frames_pages() # Extract frames and pages

    def __extract_frames_pages(self):
        self.__extract_frames()
        self.__extract_pages()
        
        # Save paths to JSON for future loading integrity checks
        self.__save_dataset_to_json(self.mapped_dataset)

        # Enable API
        logging.info("Data preparation complete.")
        self.loaded = True
        return

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
            concatenated_video_path = os.path.join(self.videos_dir, f"object_{self.tmp_index}_concatenated_video.mp4")
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
            manual_url = manuals[manual_num]
            pdf_path = os.path.join(self.manuals_dir, instruction_manual_filename)
            self.__download_manual(manual_url, pdf_path)

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

    def __save_dataset_to_json(self, data):
        """Saves the dataset to a JSON file.

        Parameters
        ----------
        data : List[Dict[str, str]]
            A list of dictionaries containing the paths to the videos and manuals.
        """
        output_json_path = os.path.join(self.base_dir, self.dataset_filename)
        with open(output_json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        logging.info(f"Dataset JSON saved to {output_json_path}")

    def __compute_video_manual_mapping(self, dataset_paths, data) -> List[Dict]:
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
                    "video": dataset_paths[index]["video_url"],
                    "manual": dataset_paths[index]["pdf"],
                    "annotations": annotated_segment["segment_annotations"]
                })
            index += 1
        return mapped_dataset

    def __extract_frames(self):
        """Extracts the first and last frames from each video and saves them to the frames directory, 
        updating the annotations with the paths to the frames.
        """
        for entry in self._mapped_dataset: # Each of these is a video-manual-annotations triplet
            timestamps = []
            video_path = entry["video"]
            video_id = video_path.split("/")[-1]
            for annotation in entry["annotations"]:
                timestamps.extend(self.__select_n_frames(annotation["start_time"], annotation["end_time"], 2)) # TODO Parametrize n with settings
            
            frames = self.__extract_frames_bulk(video_path, timestamps) # Returns list of PIL images
            paths = self.__save_frames_paths(video_id, frames) # Returns list of lists of paths to the frames

            assert len(paths) == len(entry["annotations"]), "Number of extracted frames doesn't match the number of annotations"

            for annotation in entry["annotations"]:
                frames_paths = paths.pop(0)
                annotation["start_frame"] = frames_paths[0]
                annotation["end_frame"] = frames_paths[1]

    def __extract_pages(self):
        for entry in self._mapped_dataset:
            manual_page_boundaries = self.__get_manual_page_boundaries(entry)
            pages = []
            pdf_path = entry["manual"]
            pdf_id = pdf_path.split("/")[-1]
            for annotation in entry["annotations"]:
                page_index_entry = annotation["page_index"]
                page_index = page_index_entry if type(page_index_entry) is int else int(page_index_entry[1])

                tmp_pg = [page_index]
                tmp_ix = tmp_pg[0] + 1
                if tmp_ix <= manual_page_boundaries[1]:
                    tmp_pg.append(tmp_ix) # Extract correct and next page if possible
                
                pages.extend(tmp_pg)
            
            pages = set(pages)
            pages_dict = self.__extract_pages_bulk(pdf_path, pages)

            paths = self.__save_pages_paths(pdf_id, pages_dict)

            for annotation in entry["annotations"]:
                page_ix = annotation["page_index"]
                annotation["page_path"] = paths[page_ix]
                if page_ix + 1 in pages_dict:
                    annotation["next_page_path"] = paths[page_ix + 1]

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

    def __extract_frames_bulk(self, video_path, timestamps) -> List[Image]:
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

    def __extract_pages_bulk(self, manual_path, pages) -> Dict[int, Image]:
        """Extracts pages from a PDF at the specified indices and returns them as PIL images.

        Parameters
        ----------
        manual_path : str
            The path to the PDF file.
        
        pages : List[int]
            A list of page indices to extract.

        Returns
        -------
        List[PIL.Image]
            A list of PIL images representing the extracted pages.
        """
        pages_dict = {}
        pdf = pdfium.PdfDocument(manual_path)
        for page_idx in pages:
            if type(page_idx) is list:
                page_idx = page_idx[0]
            page = pdf[page_idx - 1] # Convert to zero-based
            image = page.render(scale=.5).to_pil()
            pages_dict[page_idx] = image

        return pages_dict

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
        obj_id, seg_id = self.__extract_x_y(video_id)
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

    def __save_pages_paths(self, pdf_id, pages):
        obj_id, seg_id = self.__extract_x_y(pdf_id)
        folder_name = f"object_{obj_id}_manual_{seg_id}"

        save_dir = os.path.join(self.pdf_images_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        paths = {}

        for ix, page in pages.items():
            page_filename = f"page_{ix}.jpg"
            page_path = os.path.join(save_dir, page_filename)
            page.save(page_path)
            paths[ix] = page_path

        return paths

    def __get_manual_page_boundaries(self, entry) -> Tuple[int, int]:
        """Extracts the min and max page from the annotations of an instruction manual.

        Parameters
        ----------
        entry : Dict
            The dataset entry containing annotations.

        Returns
        -------
        Tuple[int, int]
            A tuple containing the minimum and maximum page indices which can be used for this manual.
        """
        annotations = entry["annotations"]

        pages = []
        for annotation in annotations:
            if type(annotation["page_index"]) is list:
                page = int(annotation["page_index"][1])
            else:
                page = int(annotation["page_index"])
            pages.append(page)
            
        return min(pages), max(pages)

    def __extract_x_y(self, file_path) -> Tuple[int, int]:
        """Extracts the object and segment IDs from the file path.

        Parameters
        ----------
        file_path : str
            The path to the file.
        
        Returns
        -------
        Tuple[int, int]
            A tuple containing the object and segment IDs.
        """
        pattern = r'object_(\d+)_(video_segment|instruction_manual)_(\d+)\.(mp4|pdf)'
        file_name = Path(file_path).name
        match = re.search(pattern, file_name)
        if match:
            x = int(match.group(1))
            y = int(match.group(3))
            return (x, y)
        else:
            raise ValueError(f"Filename '{file_name}' doesn't match the expected pattern.")

