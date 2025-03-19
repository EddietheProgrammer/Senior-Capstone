import numpy as np
from typing import List, Tuple

def coco_2_h36m(x: np.ndarray[float, float]) -> np.ndarray[float, float]:
    """
    Input x (1 x KP x xy)

    COCO Format:
    0-Nose, 1-Leye, 2-Reye, 3-Lear, 4-Rear, 5-Lshoulder, 6-Rshoulder, 7-Lelbow, 8-Relbow, 9-Lwrist, 10-Rwrist,
    11-Lhip, 12-Rhip, 13-Lknee, 14-Rknee, 15-Lankle, 16-Rankle

    H36M Format:
    0-root, 1-Rhip, 2-Rknee, 3-Rankle, 4-Lhip, 5-Lknee, 6-Lankle, 7-belly, 8-Neck, 9-Nose, 10-Head, 11-Lshoulder,
    12-Lelbow, 13-Lwrist, 14-Rshoulder, 15-Relbow, 16-Rwrist
    """
    # Shape is (1, 17, 2) (Index, Keypoints, X and Y coordinate)
    y = np.zeros(x.shape)

    y[:, 0, :] = (x[:, 11, :]) + (x[:, 12, :]) * 0.5 # Halves the shoulders to get center
    y[:, 1, :] = x[:, 12, :]
    y[:, 2, :] = x[:, 14, :]
    y[:, 3, :] = x[:, 16, :]
    y[:, 4, :] = x[:, 11, :]
    y[:, 5, :] = x[:, 13, :]
    y[:, 6, :] = x[:, 15, :]
    y[:, 7, :] = (x[:, 0, :] + x[:, 8, :]) * 0.5
    y[:, 8, :] = (x[:, 5, :] + x[:, 6, :]) * 0.5
    y[:, 9, :] = x[:, 0, :]
    y[:, 10, :] = (x[:, 1, :] + x[:, 2, :]) * 0.5
    y[:, 11, :] = x[:, 5, :]
    y[:, 12, :] = x[:, 7, :]
    y[:, 13, :] = x[:, 9, :]
    y[:, 14, :] = x[:, 6, :]
    y[:, 15, :] = x[:, 8, :]
    y[:, 16, :] = x[:, 10, :]

    return y


# Alphapose Format
# x1, y1, c1 – x,y coordinate of body part and the confidence score
# score – the yolo model score

def convert_2_alphapose(coordinates: List[Tuple[float, float]], 
                        body_scores: List[float], 
                        classifier_score: int) -> dict:
    frame_dict = {}

    coord_list = []
    for item, score in zip(coordinates, body_scores):
        x,y = item

        coord_list.extend([float(x), float(y), float(score)])

    frame_dict['image_id'] = "Pitcher"
    frame_dict['category_id'] = 1
    frame_dict['keypoints'] = coord_list
    frame_dict['score'] = classifier_score

    return frame_dict