import cv2
from baseballcv.functions import LoadTools
from ultralytics import YOLO
from rtmlib import PoseTracker, Body, draw_skeleton
from utils import coco_2_h36m, convert_2_alphapose
import numpy as np
import json


tools = LoadTools()

device = 'cpu'  # cpu, cuda
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

model = YOLO(tools.load_model('phc_detector', model_type='YOLO'))

cap = cv2.VideoCapture('assets/test.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 40
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

# out = cv2.VideoWriter('assets/test-output.mp4', fourcc, fps, (frame_width, frame_height))

tracker = PoseTracker(Body, 7, False, mode = 'performance', backend=backend, device=device)

alphapose_fmt = []

while cap.isOpened():
    read, frame = cap.read()

    if not read:
        print('Something went wrong')
        break
    
    results = model.predict(frame, device='mps', stream=True)

    for result in results:
        c_names = result.names
        for box in result.boxes:
            box_score = box.conf[0]
            if box_score > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cls = int(box.cls[0])
                class_name = c_names[cls]

                if class_name == 'pitcher':
                    pitcher_frame = frame[y1:y2, x1:x2]
                   
                    key, score = tracker(pitcher_frame)


                    img_show = frame.copy()


                    if len(key) == 2: # Only want pitcher, not catcher.
                        key = key[0].reshape(1, 17, 2)
                        score = score[0].reshape(1, 17)

                    # Remember: Draw skeleton is expecting the input to be COCO format
                    img_show = draw_skeleton(pitcher_frame, key, score, False, 0.5, line_width=3)

                    conversion = np.squeeze(coco_2_h36m(key))
                    scores = np.squeeze(score) # Bad variable name
                    flatten_conversion = [(x, y) for x, y in conversion]

                    alphapose = convert_2_alphapose(flatten_conversion, scores, box_score.item())

                    alphapose_fmt.append(alphapose)


                    cv2.circle(pitcher_frame, (int(key[0, 14, 0]), int(key[0, 14, 1])), 3, (0, 0, 255), 5) # Should be Right knee
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0 ,255, 0), 2)
                    cv2.putText(frame, f'{c_names[int(box.cls[0])]} {box.conf[0]:.2f}', 
                                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #out.write(frame)
    cv2.imshow('img', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open('test.json', 'w') as f:
    alph_json = json.dumps(alphapose_fmt)
    f.write(alph_json)

