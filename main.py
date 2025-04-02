from baseballcv.functions import LoadTools
from ultralytics import YOLO
from rtmlib import PoseTracker, Body
from threeD_estimation import ThreeDEstimator

tools = LoadTools()
pose_model = YOLO(tools.load_model('phc_detector', model_type='YOLO')) # You can use whatever pose model you want, for this though I recommend baseballcv, it's awesome
tracker = PoseTracker(Body, 7, False, mode='performance', backend='onnxruntime', device='mps')

estimator = ThreeDEstimator('assets/test.mp4', 'test')

estimator.extract_2D_coordinates(pose_model, tracker)
estimator.write_2D_output(pose_model, tracker, True)
estimator.write_3D_output()
