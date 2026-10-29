from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = Path(__file__).parent / "people_on_street.mp4"
MODEL_PATH = "yolov8n.pt"
PERSON_CLASS_ID = 0  # COCO class 0 = person
WINDOW_NAME = "Object Tracking"
TEXT_POSITION = (10, 30)
TEXT_COLOR = (0, 255, 0)  # green (BGR)


def count_unique_people(video_path: Path, model_path: str) -> int:
    """Track people in a video and count how many unique individuals appear.

    Returns the total number of unique track IDs observed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {video_path}")

    model = YOLO(model_path)
    unique_ids: set[int] = set()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # end of video stream (or read failure)

            results = model.track(
                frame, classes=[PERSON_CLASS_ID], persist=True, verbose=False
            )
            annotated_frame = results[0].plot()

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                unique_ids.update(boxes.id.int().cpu().tolist())

            cv2.putText(
                annotated_frame,
                f"Count: {len(unique_ids)}",
                TEXT_POSITION,  # position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1,  # font scale
                TEXT_COLOR,
                2,  # thickness
            )
            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return len(unique_ids)


if __name__ == "__main__":
    total = count_unique_people(VIDEO_PATH, MODEL_PATH)
    print(f"Total unique people counted: {total}")
