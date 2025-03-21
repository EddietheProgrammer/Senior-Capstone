import numpy as np
from typing import List, Tuple, Dict, Any
import cv2

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

    y[:, 0, :] = (x[:, 11, :] + x[:, 12, :]) * 0.5 # Halves the hips to get center
    y[:, 1, :] = x[:, 12, :]
    y[:, 2, :] = x[:, 14, :]
    y[:, 3, :] = x[:, 16, :]
    y[:, 4, :] = x[:, 11, :]
    y[:, 5, :] = x[:, 13, :]
    y[:, 6, :] = x[:, 15, :]
    y[:, 8, :] = (x[:, 5, :] + x[:, 6, :]) * 0.5
    y[:, 7, :] = (y[:, 0, :] + y[:, 8, :]) * 0.5 # Center this with the neck and root
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
                        classifier_score: float) -> Dict[str, Any]:
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

def draw(img, keypoints, keypoint_info, skeleton_info, radius=6, line_width=2):
    assert len(keypoints.shape) == 2, "Invalid shape for keypoints, must be 2."

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']

        kpt = keypoints[i]

        img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius),
                             kpt_color, -1)

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        link_color = ske_info['color']
        kpt0 = keypoints[pt0]
        kpt1 = keypoints[pt1]

        img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])),
                        (int(kpt1[0]), int(kpt1[1])),
                        link_color,
                        thickness=line_width)
    return img
