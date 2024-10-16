import cv2
import numpy as np
from histogram import main

image = cv2.imread("train/train/skin_cancer/skin_cancer_15.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(3,3),0)

def autoCanny(blur, sigma = 0.8):
    median = np.median(blur)
    lower = int(max(0,(1.0-sigma)*median))
    upper = int(min(255,(1.0+sigma)*median))
    canny = cv2.Canny(blur, lower, upper)
    return canny

auto = autoCanny(blur)

auto= cv2.resize(auto, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
blur= cv2.resize(blur, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

cv2.imshow("blurred image",blur)
cv2.imshow("auto",auto)

cv2.waitKey(0)
cv2.destroyAllWindows()