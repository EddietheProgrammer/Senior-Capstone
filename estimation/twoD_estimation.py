import cv2
from cv2.typing import MatLike
import os
from baseballcv.functions import LoadTools
from ultralytics import YOLO
from rtmlib import PoseTracker, Body
from estimation.utils import coco_2_h36m, convert_2_alphapose, draw
import numpy as np
import json
from estimation.h36m import h36m
from tqdm import tqdm
from typing import List, Dict, Any

# TODO: Fix the video writer function. It is for some reason not showing the annotations.
class TwoDEstimator:
    """
    Class that estimates the 2D poses of MLB Pitchers. 2 Main Functions
    1. Write Frame: Purpose of this is to visualize the video and save it as mp4 for you to show your friends.
    2. Write 2d Json: Purpose of this is to convert the poses to H36M format, which is then fed into the 3D model.
    """

    def __init__(self, video_path: str, detector: YOLO, tracker: PoseTracker) -> None:
        self.video_path = video_path
        self.phc_model = detector
        self.tracker = tracker
        self.skeleton_dict = eval('h36m')

    def output_frames(self, frame: MatLike, keypoints: np.ndarray, skeleton_dict: Dict[str, Any]) -> MatLike:
        """
        Function that takes in the frames and keypoints to draw the skeleton.

        Args:
            frame (MatLike): The processed frame of the video.
            keypoints (ndarray): The keypoints of each bodypart, presented in h36m format.
            skeleton_dict (dict): The skeleton dictionary, in h36m format, of each body part and drawn line.

        Returns:
            img (MatLike): The processed image with updated annotations for the skeleton.
        """
        for i in range(keypoints.shape[0]):
            keypoint_info = skeleton_dict['keypoint_info']
            skeleton_info = skeleton_dict['skeleton_info']
            img = draw(frame, keypoints[i], keypoint_info, skeleton_info)

        return img

    def process_frame(self, frame: MatLike, output_frame: bool = False, 
                      write_frame: bool = False) -> List[Dict[str, Any]] | MatLike:
        """
        Processes the frames and converts it from coco17 to h36m format. A YOLO model is first ran to identify
        the pitcher, then RTMPose is used to extract the body coordinates for the pitcher.

        Args:
            frame (MatLike): The processed frame of the video.
            output_frame (bool): A condition of whether you want to output the frame on the screen, default to False.

        Returns:
            alpahose_fmt (list): A list of dictionaries that are in the correct alphapose format.
        """
        alphapose_fmt = []
        results = self.phc_model.predict(frame, device='mps', stream = True)

        for result in results:
            c_names = result.names
            for box in result.boxes:
                box_score = box.conf[0]
                if box_score > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    c_idx = int(box.cls[0])
                    class_name = c_names[c_idx]

                    if class_name == 'pitcher':
                        pitcher_frame = frame[y1:y2, x1:x2]

                        key, score = self.tracker(pitcher_frame)

                        if len(key) == 2: # Only want pitcher, not catcher
                            key = key[0].reshape(1, 17, 2)
                            score = score[0].reshape(1, 17)
                        
                        key = coco_2_h36m(key)
                        conversion = np.squeeze(key)
                        score = np.squeeze(score)

                        flatten_conversion = [(x, y) for x,y in conversion]
                        alphapose = convert_2_alphapose(flatten_conversion, score, box_score.item())
                        alphapose_fmt.append(alphapose)

                        if output_frame:
                            self.output_frames(pitcher_frame, key, self.skeleton_dict)
        if write_frame:
            return frame
        else:
            return alphapose_fmt
    
    def read_frame(self) -> List[Dict[str, Any]]:
        """
        Reads in each frame, processes it to alphapose format, then returns it.

        Args:
            None YAY!!
        Returns:
            alphapose_fmt (list): A list of dictionaries that are in the correct alphapose format.
        """
        cap = cv2.VideoCapture(self.video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        alphapose_fmt = []
        
        with tqdm(total=n_frames, desc='Processing Frames For Json Output', unit='frame') as progress:
            while cap.isOpened():
                read, frame = cap.read()

                if not read:
                    print('Something went wrong. Most likely, the video ended.')
                    break

                alphapose_fmt.extend(self.process_frame(frame))

                progress.update(1)

        cap.release()
        return alphapose_fmt
    
    def write_frame(self, out_name: str, output_frame: bool = False) -> None:
        """
        Saves the annotated video as a mp4 so you can show it off to your friends.
        Make sure to have an assets/ folder. That's where the video will be saved.

        Args:
            output_path (str): The output path you want the annotated video to save. Doesn't need to have .mp4. I've done that.
            output_frame (bool): A condition of whether you want to output the frame on the screen, default to False.

        Returns:
            None
        """
        assert os.path.exists('assets/'), "Need to have an assets/ folder in your local directory."
        assert not out_name[-4:] == '.mp4', "You don't need to end it with .mp4, I've already done that."
        assert '.' not in out_name, "Please, no . in output name, also only supported extension is .mp4"

        cap = cv2.VideoCapture(self.video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 40
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        out = cv2.VideoWriter(f'assets/{out_name}.mp4', fourcc, fps, (frame_width, frame_height))

        with tqdm(total=n_frames, desc='Processing Frames For Video Writer', unit='frame') as progress:
            while cap.isOpened():
                read, frame = cap.read()

                if not read:
                    print('Something went wrong. Most likely, the video ended.')
                    break

                out.write(self.process_frame(frame, output_frame, True))
                if output_frame:
                    cv2.imshow("pitcher", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                progress.update(1)
        
        cap.release()
        cv2.destroyAllWindows()

    def write_2d_frame_json(self, json_name: str) -> None:
        """
        Writes the converted alphapose dictionary format to json. The file is written in your local directory.
        
        Hint: Don't put .json, I've already done that for you.

        Args:
            json_name (str): The json file name you want the json to save.
        
        Returns:
            None
        """
        assert not json_name[-5:] == '.json', "You don't need to specify .json, I've already done that."
        assert '.' not in json_name, "Don't use . in the json output name. Also, only supported file type is .json"

        with open(f'{json_name}.json', 'w') as f:
            alph_json = json.dumps(self.read_frame())
            f.write(alph_json)


if __name__ == '__main__':
    tools = LoadTools()
    pose_model = YOLO(tools.load_model('phc_detector', model_type='YOLO')) # You can use whatever pose model you want, for this though I recommend baseballcv, it's awesome
    tracker = PoseTracker(Body, 7, False, mode='performance', backend='onnxruntime', device='cpu')
    estimator = TwoDEstimator('assets/test.mp4', pose_model, tracker)

    estimator.write_2d_frame_json('cool')
