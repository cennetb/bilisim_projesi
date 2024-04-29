import cv2

def goster1(image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Image", image)
    return