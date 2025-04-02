import os
import numpy as np
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save
from estimation.twoD_estimation import TwoDEstimator
from scipy.signal import butter, filtfilt
from ultralytics import YOLO
from rtmlib import PoseTracker

class ThreeDEstimator:
    """
    Class that implements the 3D modeling from MotionBERT. It allows you to use the `TwoDEstimator` class
    to extract and clean the 2D estimates before feeding it into the 3D model. 

    Main Functions:
    1. extract_2D_coordinates: This extracts the 2D coordinates of a 2D video feed and write it to a json file.
    2. write_2D_output: This writes the 2D coordinates (in H36M format) into a video file, for you to show off to your friends.
    3. run_model: This returns the predicted xyz coordinate array of the 3D pose for each frame.
    4. write_3D_output: This writes the 3D coordinates into a video file for you to show off to your friends.

    In progress: Making a function that derives various biomechanical metrics.
    """
    def __init__(self, video_path: str, json_name: str) -> None:
        self.video_path = video_path
        self.json_name = json_name
        self.arguments = get_config('configs/pose3d/MB_ft_h36m_global_lite.yaml') # Where the model config lies in the file system
        self.backbone = load_backbone(self.arguments)
        self.checkpoint = torch.load('checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', map_location=lambda storage, loc: storage)
        self.checkpoint = {k.replace('module.', ''): v for k, v in self.checkpoint['model_pos'].items()}

        self.backbone.load_state_dict(self.checkpoint, strict=True)
        self.model_pos = self.backbone
        self.model_pos.eval() # Evaluates the checkpoints of the pytorch model

        self.testloader_params = {
                        'batch_size': 1,
                        'shuffle': False,
                        'num_workers': 0, # was 8
                        'pin_memory': True,
                        'prefetch_factor': None,
                        'persistent_workers': False,
                        'drop_last': False
                }
        
        self.vid =  imageio.get_reader(self.video_path, 'ffmpeg')
        self.fps_in = self.vid.get_meta_data()['fps']
        self.vid_size = self.vid.get_meta_data()['size']

    def extract_2D_coordinates(self, pose_model: YOLO, tracker: PoseTracker) -> None:
        """
        Extracts the 2D coordinates from the input file of the MLB video feed. 
        The 2D coordinates are then formatted into what MotionBERT likes and written to 
        a json file. 

        Args:
            pose_model (YOLO): The model to use for classifying objects. BaseballCV's phc_detector model suffices for this.
            The reason it's not in the model is because RTMLib needs to `__call__` a YOLO model, and putting it in a function
            creates an error. 
            tracker (PoseTracker): The tracking model to use for classifying each body part. This function is tuned more to
            RTMPose where the key points are in the COCO-17 format.

        Returns:
            None :)
        
        """
        TwoDEstimator(self.video_path, pose_model, tracker).write_2d_frame_json(self.json_name)
    
    def write_2D_output(self, pose_model: YOLO, tracker: PoseTracker, output_frame: bool = False) -> None:
        """
        Writes the 2D video feed into another .mp4 file with the annotated poses and keypoints. 

        Args:
            pose_model (YOLO): The model to use for classifying objects. BaseballCV's phc_detector model suffices for this.
            The reason it's not in the model is because RTMLib needs to `__call__` a YOLO model, and putting it in a function
            creates an error. 
            tracker (PoseTracker): The tracking model to use for classifying each body part. This function is tuned more to
            RTMPose where the key points are in the COCO-17 format.
            output_frame (bool): Whether you want to output the video while it's writing or not. Default to not.

        Returns:
            None :)
        """
        TwoDEstimator(self.video_path, pose_model, tracker).write_frame(out_name='test-output', output_frame=output_frame)

    def _butterworth_filter(self, data: np.ndarray, cutoff_freq: int, fs: float, order: int=2) -> np.ndarray:
        """
        Creates the a lowpass filter for data that's fed into it. It's a 2nd order filter to maintain integrity
        of the model. 

        Args:
            data (np.ndarray): The data you want to filter in an array format, shape should be 1 X N.
            cutoff_freq (int): The cutoff in HZ you want the filter to consider high pass.
            fs (float): The frequency of the input.
            order (int): The nth order you want the frequency to be derived to. Default to 2. DON'T CHANGE IT!!!

        Returns:
            np.ndarray: An 1 X N dimension array of the filtered data.
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def _apply_lowpass_butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the `_butterworth_filter` function into the input array for each body part and each xyz coordinate. 
        The shape should be (N X 17 X 3) where N is the number of frames, 17 is the number of keypoints, and 3 are the 
        xyz coordinates.

        Args:
            data (np.ndarray): The input data for the filter. It's the predicted 3D output for each frame's bodypart.

        Returns:
            np.ndarray: The array of the data with the butterworth filter applied.
        """
        filt_results = []

        for i in range(17): # Iterate over each bodypart
            part = data[:, i, :]
            
            x, y, z = part[:, 0], part[:, 1], part[:, 2]
        
            x = self._butterworth_filter(x, 5, self.fps_in)
            y = self._butterworth_filter(y, 5, self.fps_in)
            z = self._butterworth_filter(z, 5, self.fps_in)

            filt_part = np.stack([x, y , z], axis=1)
            filt_results.append(filt_part)

        return np.stack(filt_results, axis=1)


    def run_model(self) -> np.ndarray:
        """
        Runs the 3D model on the json input.

        Returns:
            np.ndarray: An array of the predicted xyz coordinate for each bodypart in each frame. The predictions
            are also filterd through with a lowpass butterworth filter.
        """
        # TODO: Add a feature to convert it to pixels

        wild_dataset = WildDetDataset(f'{self.json_name}.json', clip_len=243,  scale_range=[1,1], focus=None)
        test_loader = DataLoader(wild_dataset, **self.testloader_params)

        results = []
        with torch.no_grad():
            for batch_input in tqdm(test_loader):
                N, T = batch_input.shape[:2]

                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos = self.model_pos(batch_input_flip)

                predicted_3d_pos[:, 0, 0, 2] = 0

                results.append(predicted_3d_pos.cpu().numpy())


        results = np.hstack(results)
        results = np.concatenate(results)


        results = self._apply_lowpass_butterworth_filter(results)

        return results
    
    # TODO: Write a function that uses the 3D model outputs to derive biomechanical data.

    def write_3D_output(self, file_name: str = '3D-output'):
        """
        Writes the 3D video feed into a .mp4 file using the coordinates derived from `run_model`. 
        The video should be flipped for LHP and RHP to make it look like catcher's view.

        Args:
            file_name (str): The desired file name for the output. It's put in the assets/3D-outputs/
            folder in your file directory. You can call it whatever you want (e.g. '3D-output'), which is
            the default.

        Returns:
            None :)
        """
        results = self.run_model()
        render_and_save(results, f'%s/{file_name}.mp4' % ('assets/3D-outputs/'), keep_imgs=False, fps = self.fps_in)