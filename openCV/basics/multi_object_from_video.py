from pathlib import Path

import cv2
from ultralytics import YOLO

path = Path(__file__).parent
cap = cv2.VideoCapture(str(path / "people_on_street.mp4"))

model = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()

    # 0 = people
    results = model(frame, classes=[0])

    annotated_frame = results[0].plot()

    cv2.imshow("Annotated video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
