import cv2
import numpy as np

canvas = np.zeros((512, 512, 3), dtype=np.uint8)

cv2.line(canvas, (0, 0), (512, 512), (0, 255, 0), 5)
cv2.rectangle(canvas, (100, 100), (400, 400), (255, 0, 0), 2)
cv2.putText(canvas, "OpenCV", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()