import cv2

def buyut(image):
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    return image