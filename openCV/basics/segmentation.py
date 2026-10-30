from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = Path(__file__).parent / "people_on_street.mp4"
MODEL_PATH = Path(__file__).parent / "yolov8n-seg.pt"

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, classes=[0], persist=True, verbose=False)
    annotated_frame = frame.copy()

    for r in results:
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            # masks.xy: per-person polygons already in original-frame pixels
            polygons = r.masks.xy
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()

            for polygon, box, person_id in zip(polygons, boxes, ids):
                x1, y1, _, _ = box.astype(int)
                cv2.polylines(
                    annotated_frame,
                    [polygon.astype(np.int32)],
                    isClosed=True,
                    color=(0, 0, 255),
                    thickness=2,
                )
                cv2.putText(
                    annotated_frame,
                    f"ID: {person_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

    cv2.imshow("Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
