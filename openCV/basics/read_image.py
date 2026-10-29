from pathlib import Path

import cv2

img = cv2.imread(str(Path(__file__).parent / 'banana.jpg'))
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()