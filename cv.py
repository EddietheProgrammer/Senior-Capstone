from baseballcv.functions import LoadTools
from ultralytics import YOLO

tools = LoadTools()
# Load pitcher-catcher-batter model

m = tools.load_model('phc_detector', model_type='YOLO')

model = YOLO(m)

pred = model.predict('assets/test.jpeg', project = '../Senior Capstone/assets', save = True, name='output')

