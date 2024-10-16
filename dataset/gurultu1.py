import cv2
import numpy as np
import matplotlib.image as mpimg

image = cv2.imread('./dataset/train/train/skin_cancer/skin_cancer_10.jpg')

kernel = np.ones((10,10),np.uint8)
#image = cv2.morphologyEx(image, cv2)

# dilation = cv2.dilate(image,kernel,iterations=1)
# erosion = cv2.erode(dilation,kernel,iterations=1)
#
# cv2.imshow('image', image)
# cv2.imshow('dilation', dilation)
# cv2.imshow('erode', erosion)

#opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)

gradyan = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradyan', gradyan)

cv2.waitKey(0)
cv2.destroyAllWindows()