from baseballcv.functions import LoadTools
from ultralytics import YOLO
import cv2

tools = LoadTools()

# Load pitcher-catcher-batter model
model = YOLO(tools.load_model('phc_detector', model_type='YOLO'))

cap = cv2.VideoCapture('assets/test.mp4')

while cap.isOpened():
    read, frame = cap.read()

    if not read:
        print("Something went wrong")
        break
    results = model.predict(frame)

    for result in results:
        classes_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cls = int(box.cls[0])
                class_name = classes_names[cls]

                if class_name == 'pitcher':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0 ,255, 0), 2)
                    cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', 
                                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Pitcher", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#pred = model.predict('assets/test.mp4', project = '../Senior Capstone/assets', save = True, name='output')

