from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = Path(__file__).parent / "people_on_street.mp4"
MODEL_PATH = Path(__file__).parent / "yolov8n.pt"
MAX_MISSING_FRAMES = 60


model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO_PATH}")

id_map: dict[int, int] = {}
next_id = 1

trail: defaultdict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))
appear: defaultdict[int, int] = defaultdict(int)
last_seen: dict[int, int] = {}
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    res = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated_frame = frame.copy()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.int().cpu().tolist()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            appear[oid] += 1
            last_seen[oid] = frame_idx

            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))

                pts = trail[oid]
                for i in range(1, len(pts)):
                    cv2.line(annotated_frame, pts[i - 1], pts[i], (0, 255, 255), 2)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"ID: {sid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

    # Drop per-track state for IDs that disappeared (id_map is kept so a
    # returning track retains its display ID).
    for oid in [
        o for o, seen in last_seen.items() if frame_idx - seen > MAX_MISSING_FRAMES
    ]:
        del last_seen[oid]
        appear.pop(oid, None)
        trail.pop(oid, None)

    cv2.imshow("People with Trail", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
