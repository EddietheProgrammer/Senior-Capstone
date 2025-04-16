from baseballcv.functions import LoadTools
from ultralytics import YOLO
from rtmlib import PoseTracker, Body
from threeD_estimation import ThreeDEstimator
import pandas as pd
import os

videos = os.listdir('assets/videos')

tools = LoadTools()
pose_model = YOLO(tools.load_model('phc_detector', model_type='YOLO')) # You can use whatever pose model you want, for this though I recommend baseballcv, it's awesome
tracker = PoseTracker(Body, 7, False, mode='performance', backend='onnxruntime', device='mps')

kinematics_data = {
    'game_pk': [],
    'play_id': [],
    'elbow_flexion': [],
    'shoulder_abduction': [],
    'knee_flexion': []
}

for video in videos[:2]:
    game_pk, play_id = video.split('_')
    play_id, _ = play_id.split('.')

    estimator = ThreeDEstimator(os.path.join('assets','videos', video), 'test') # Doesn't have to be called 'test', but I'm doing it for simplicity
    estimator.extract_2D_coordinates(pose_model, tracker)

    elb, shoul, knee = estimator.derive_kinematics()

    kinematics_data['game_pk'].append(game_pk)
    kinematics_data['play_id'].append(play_id)
    kinematics_data['elbow_flexion'].append(elb)
    kinematics_data['shoulder_abduction'].append(shoul)
    kinematics_data['knee_flexion'].append(knee)

    estimator.clear_2D_json()


df = pd.DataFrame(kinematics_data)
print(df)


